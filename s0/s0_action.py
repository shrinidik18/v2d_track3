"""Reference-free hand action: the policy moves its own wrist and finger targets (integrated deltas), so the reconstructed
hand trajectory plays no part in control. Selected with `action=relative_hand_pose` (config in /env/cfg/action/).

Per control step and hand: target_pos += delta_pos, target_quat = target_quat * rpy(delta_rot), finger_targets += delta,
with delta = EMA(clamp(a, -1, 1) * max_step). Targets start from the robot's measured pose after every reset and fingers
are clamped to their joint limits. The existing wrist PD controller tracks the targets (gravity compensation is on).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import warp as wp

from flash_chord.runtime.actions.hand_pose import ActionLayout
from flash_chord.runtime.control import WristController
from flash_chord.runtime.delay import TargetDelayConfig
from flash_chord.utils.quat import quat_mul_xyzw, quat_to_rotvec


@dataclass(frozen=True)
class RelativeHandPoseConfig:
    wrist_pos_step: float = 0.03          # m per control step at |a| = 1 (0.6 m/s at 20 Hz)
    wrist_ori_step: float = 0.10          # rad per step
    finger_joint_step: float = 0.10       # rad per step
    ema: float = 0.3
    wrist_kp: float = 25.0
    wrist_tau_max: float = 45.0
    delay: TargetDelayConfig = field(default_factory=TargetDelayConfig)
    # accepted for compatibility with recipe overrides; this action always reads normalized [-1, 1] input
    input_mode: Literal["raw", "normalized"] = "normalized"
    normalized_mapping: Literal["linear", "rational"] = "linear"

    def build(self, scene, device=None):
        return RelativeHandPoseAction.build(scene, self, device=device)


@wp.kernel
def _process_relative(
    action: wp.array(dtype=wp.float32), joint_q: wp.array(dtype=wp.float32), num_joint_q: int, num_joint_dof: int,
    num_hands: int, action_dim: int, processed_dim: int, total_fingers: int,
    hand_action_starts: wp.array(dtype=wp.int32), hand_processed_starts: wp.array(dtype=wp.int32),
    hand_finger_starts: wp.array(dtype=wp.int32), finger_counts: wp.array(dtype=wp.int32),
    wrist_pos_q_ids: wp.array(dtype=wp.int32), wrist_quat_q_ids: wp.array(dtype=wp.int32), finger_q_ids: wp.array(dtype=wp.int32),
    wrist_pos_dof_ids: wp.array(dtype=wp.int32), wrist_ori_dof_ids: wp.array(dtype=wp.int32), finger_dof_ids: wp.array(dtype=wp.int32),
    limit_lower: wp.array(dtype=wp.float32), limit_upper: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32), ema: float,
    needs_init: wp.array(dtype=wp.int32),
    raw_action: wp.array(dtype=wp.float32), filtered_action: wp.array(dtype=wp.float32),
    target_pos: wp.array(dtype=wp.vec3), target_quat: wp.array(dtype=wp.quat), target_finger: wp.array(dtype=wp.float32),
    processed_target: wp.array(dtype=wp.float32), joint_target: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32), action_rate_l2: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    ab = w * action_dim
    l2 = float(0.0)
    rate = float(0.0)
    for i in range(action_dim):
        v = wp.clamp(action[ab + i], -1.0, 1.0)
        d = v - raw_action[ab + i]
        l2 += v * v
        rate += d * d
        raw_action[ab + i] = v
        filtered_action[ab + i] = ema * filtered_action[ab + i] + (1.0 - ema) * v * step[i]
    action_l2[w] = l2
    action_rate_l2[w] = rate
    qb = w * num_joint_q
    db = w * num_joint_dof
    init = needs_init[w]
    for h in range(num_hands):
        th = w * num_hands + h
        if init != 0:
            target_pos[th] = wp.vec3(joint_q[qb + wrist_pos_q_ids[h * 3]], joint_q[qb + wrist_pos_q_ids[h * 3 + 1]], joint_q[qb + wrist_pos_q_ids[h * 3 + 2]])
            o = qb + wrist_quat_q_ids[h]
            target_quat[th] = wp.normalize(wp.quat(joint_q[o], joint_q[o + 1], joint_q[o + 2], joint_q[o + 3]))
        a0 = ab + hand_action_starts[h]
        p = target_pos[th] + wp.vec3(filtered_action[a0], filtered_action[a0 + 1], filtered_action[a0 + 2])
        q = wp.normalize(quat_mul_xyzw(target_quat[th], wp.quat_rpy(filtered_action[a0 + 3], filtered_action[a0 + 4], filtered_action[a0 + 5])))
        target_pos[th] = p
        target_quat[th] = q
        pb = w * processed_dim + hand_processed_starts[h]
        processed_target[pb] = p[0]
        processed_target[pb + 1] = p[1]
        processed_target[pb + 2] = p[2]
        processed_target[pb + 3] = q[3]
        processed_target[pb + 4] = q[0]
        processed_target[pb + 5] = q[1]
        processed_target[pb + 6] = q[2]
        rv = quat_to_rotvec(q)
        for axis in range(3):
            joint_target[db + wrist_pos_dof_ids[h * 3 + axis]] = p[axis]
            joint_target[db + wrist_ori_dof_ids[h * 3 + axis]] = rv[axis]
        fs = hand_finger_starts[h]
        for f in range(finger_counts[h]):
            fi = w * total_fingers + fs + f
            if init != 0:
                target_finger[fi] = joint_q[qb + finger_q_ids[fs + f]]
            dof = finger_dof_ids[fs + f]
            t = wp.clamp(target_finger[fi] + filtered_action[a0 + 6 + f], limit_lower[db + dof], limit_upper[db + dof])
            target_finger[fi] = t
            processed_target[pb + 7 + f] = t
            joint_target[db + dof] = t
    needs_init[w] = 0


@wp.kernel
def _reset_relative(reset_mask: wp.array(dtype=wp.int32), action_dim: int, needs_init: wp.array(dtype=wp.int32),
                    raw_action: wp.array(dtype=wp.float32), filtered_action: wp.array(dtype=wp.float32),
                    action_l2: wp.array(dtype=wp.float32), action_rate_l2: wp.array(dtype=wp.float32)):
    w = wp.tid()
    if reset_mask[w] == 0:
        return
    for i in range(action_dim):
        raw_action[w * action_dim + i] = 0.0
        filtered_action[w * action_dim + i] = 0.0
    action_l2[w] = 0.0
    action_rate_l2[w] = 0.0
    needs_init[w] = 1


@dataclass
class RelativeHandPoseAction:
    layout: ActionLayout
    wrist: WristController
    world_count: int
    num_joint_q: int
    arrays: dict
    raw_action: wp.array
    filtered_action: wp.array
    processed_target: wp.array
    joint_target: wp.array
    action_l2: wp.array
    action_rate_l2: wp.array
    ema: float
    limits: tuple = ()
    input_mode: str = "normalized"
    input_mapping: str = "linear"

    @classmethod
    def build(cls, scene, config: RelativeHandPoseConfig, device=None):
        W = scene.world_count
        num_joint_dof = scene.model.joint_dof_count // W
        num_joint_q = scene.model.joint_coord_count // W
        layout = ActionLayout.from_embodiment(scene.layout, sides=("right", "left"), num_joint_dof=num_joint_dof)
        pos_q, quat_q, fin_q = [], [], []
        for side in layout.sides:
            hand = scene.layout.hand(side)
            pos_q.extend(hand.wrist_pos_q_ids); quat_q.append(hand.wrist_orient_q_id); fin_q.extend(hand.finger_q_ids)
        step = np.empty(layout.action_dim, np.float32)
        for start, n in zip(layout.hand_action_starts, layout.finger_counts):
            step[start:start + 3] = config.wrist_pos_step
            step[start + 3:start + 6] = config.wrist_ori_step
            step[start + 6:start + 6 + n] = config.finger_joint_step
        i32 = lambda v: wp.array(np.asarray(v, np.int32), dtype=wp.int32, device=device)  # noqa: E731
        z = lambda n, dt=wp.float32: wp.zeros(n, dtype=dt, device=device)  # noqa: E731
        arrays = dict(
            hand_action_starts=i32(layout.hand_action_starts), hand_processed_starts=i32(layout.hand_processed_starts),
            hand_finger_starts=i32(layout.hand_finger_starts), finger_counts=i32(layout.finger_counts),
            wrist_pos_q_ids=i32(pos_q), wrist_quat_q_ids=i32(quat_q), finger_q_ids=i32(fin_q),
            wrist_pos_dof_ids=i32(layout.wrist_pos_dof_ids), wrist_ori_dof_ids=i32(layout.wrist_ori_dof_ids),
            finger_dof_ids=i32(layout.finger_dof_ids), step=wp.array(step, dtype=wp.float32, device=device),
            needs_init=wp.ones(W, dtype=wp.int32, device=device), target_pos=z(W * layout.num_hands, wp.vec3),
            target_quat=wp.array(np.tile([0, 0, 0, 1], (W * layout.num_hands, 1)).astype(np.float32), dtype=wp.quat, device=device),
            target_finger=z(W * layout.total_fingers))
        return cls(layout=layout,
                   wrist=WristController.from_layout(scene.layout, scene.model, W, kp=config.wrist_kp, tau_max=config.wrist_tau_max, device=device),
                   world_count=W, num_joint_q=num_joint_q, arrays=arrays, raw_action=z(W * layout.action_dim),
                   filtered_action=z(W * layout.action_dim), processed_target=z(W * layout.processed_dim),
                   joint_target=z(W * num_joint_dof), action_l2=z(W), action_rate_l2=z(W), ema=config.ema,
                   limits=(scene.model.joint_limit_lower, scene.model.joint_limit_upper))

    @property
    def action_dim(self): return self.layout.action_dim

    @property
    def processed_dim(self): return self.layout.processed_dim

    @property
    def sides(self): return self.layout.sides

    @property
    def normalized_mapping(self): return "linear"

    @property
    def input_scale_values(self): return tuple(1.0 for _ in range(self.layout.action_dim))

    @property
    def block_names(self):
        return tuple(n for s in self.sides for n in (f"{s}_wrist_position_delta_normalized", f"{s}_wrist_orientation_delta_normalized",
                                                     f"{s}_finger_joint_delta_normalized"))

    @property
    def block_ranges(self):
        r = []
        for start, n in zip(self.layout.hand_action_starts, self.layout.finger_counts):
            r.extend(((start, start + 3), (start + 3, start + 6), (start + 6, start + 6 + n)))
        return tuple(r)

    def process(self, action, timestep, state=None):
        if state is None:
            raise ValueError("the relative hand action needs the live state")
        a, L = self.arrays, self.layout
        wp.launch(_process_relative, dim=self.world_count, inputs=[
            action, state.joint_q, self.num_joint_q, L.num_joint_dof, L.num_hands, L.action_dim, L.processed_dim, L.total_fingers,
            a["hand_action_starts"], a["hand_processed_starts"], a["hand_finger_starts"], a["finger_counts"],
            a["wrist_pos_q_ids"], a["wrist_quat_q_ids"], a["finger_q_ids"], a["wrist_pos_dof_ids"], a["wrist_ori_dof_ids"],
            a["finger_dof_ids"], self.limits[0], self.limits[1], a["step"], self.ema, a["needs_init"],
            self.raw_action, self.filtered_action, a["target_pos"], a["target_quat"], a["target_finger"],
            self.processed_target, self.joint_target, self.action_l2, self.action_rate_l2])

    def prepare_control(self, control):
        self.wrist.write_position_targets(control, self.joint_target)

    def apply_control(self, state, control):
        self.wrist.apply_wrist_effort(state, control, self.joint_target)

    def reset(self, reset_mask):
        a = self.arrays
        wp.launch(_reset_relative, dim=self.world_count, inputs=[reset_mask, self.layout.action_dim, a["needs_init"],
                  self.raw_action, self.filtered_action, self.action_l2, self.action_rate_l2])
