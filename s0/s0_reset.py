"""Reset policy with grasp-acquisition resets and object-perturbed recovery resets (config `reset=s0_pickup`).

The curriculum's per-stage `reset_to_first_frame_probability` p selects the mode:
  p <= 0.1  FlashCHORD default: uniform reference frame, frame 0 with probability p once the VOC is below its threshold.
  p >  0.1  acquisition mode: with probability p the world resets for grasp acquisition (half at frame 0, half at a frame
            0.25-0.5 s before a pickup onset of the object trajectory, `pickup_reset_frames` in the extras file);
            otherwise a uniform frame.
Recovery: a uniform-frame reset (not frame 0, not acquisition) is perturbed with probability `perturb_probability`: every
object's pose is offset in the horizontal plane (|dx|, |dy| <= perturb_position_m) and rotated about the vertical axis
(|yaw| <= perturb_yaw_deg). Hands stay in the reference pose. Applied right after FlashCHORD's reset (graph-safe).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

import flash_chord.envs.base as _envs_base
from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

_REG: dict = {}


@dataclass(frozen=True)
class S0ResetConfig(ReferenceResetConfig):
    extras_path: str = ""
    perturb_probability: float = 0.2
    perturb_position_m: float = 0.02
    perturb_yaw_deg: float = 10.0

    def build(self, world_count: int, num_frames: int, device=None):
        return S0Reset.build(world_count, num_frames, config=self, device=device)


@wp.kernel
def _sample(reset_mask: wp.array(dtype=wp.int32), num_frames: int, voc_decay_steps: int, always_first_frame: int,
            first_frame_probability: wp.array(dtype=wp.float32), first_frame_voc_threshold: float,
            reset_finger_openness: float, seed: int, curriculum_voc_scale: wp.array(dtype=wp.float32),
            pickup_frames: wp.array(dtype=wp.int32), n_pickup: int, perturb_probability: float, perturb_pos: float,
            perturb_yaw: float, num_objects: int,
            reset_count: wp.array(dtype=wp.int32), reset_frame: wp.array(dtype=wp.int32), finger_scale: wp.array(dtype=wp.float32),
            steps_since_reset: wp.array(dtype=wp.int32), applied_voc_scale: wp.array(dtype=wp.float32),
            perturb_flag: wp.array(dtype=wp.int32), perturb_xyyaw: wp.array(dtype=wp.vec3)):
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    rng = wp.rand_init(seed, world + reset_count[world] * 104729)
    p = first_frame_probability[0]
    frame = int(0)
    uniform = int(0)
    if always_first_frame == 0 and num_frames > 1:
        frame = int(wp.randf(rng) * float(num_frames - 1))
        uniform = 1
        if p > 0.1 + 1.0e-6:
            if wp.randf(rng) < p:
                uniform = 0
                frame = 0
                if n_pickup > 0 and wp.randf(rng) < 0.5:
                    frame = pickup_frames[wp.min(int(wp.randf(rng) * float(n_pickup)), n_pickup - 1)]
        elif curriculum_voc_scale[0] < first_frame_voc_threshold:
            if wp.randf(rng) < p:
                frame = 0
                uniform = 0
    reset_frame[world] = frame
    finger_scale[world] = wp.randf(rng) * reset_finger_openness
    steps_since_reset[world] = 0
    if voc_decay_steps == 0:
        applied_voc_scale[world] = curriculum_voc_scale[0]
    else:
        applied_voc_scale[world] = 1.0
    perturb_flag[world] = 0
    if uniform == 1 and frame > 0 and wp.randf(rng) < perturb_probability:
        perturb_flag[world] = 1
        for o in range(num_objects):
            perturb_xyyaw[world * num_objects + o] = wp.vec3((2.0 * wp.randf(rng) - 1.0) * perturb_pos,
                                                             (2.0 * wp.randf(rng) - 1.0) * perturb_pos,
                                                             (2.0 * wp.randf(rng) - 1.0) * perturb_yaw)
    reset_count[world] += 1


@wp.kernel
def _clear(reset_mask: wp.array(dtype=wp.int32), perturb_flag: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if reset_mask[w] != 0:
        perturb_flag[w] = 0


@wp.kernel
def _perturb(joint_q: wp.array(dtype=wp.float32), per_world_q: int, reset_mask: wp.array(dtype=wp.int32),
             perturb_flag: wp.array(dtype=wp.int32), perturb_xyyaw: wp.array(dtype=wp.vec3),
             object_root_q_ids: wp.array(dtype=wp.int32), num_objects: int):
    w = wp.tid()
    if reset_mask[w] == 0 or perturb_flag[w] == 0:
        return
    for o in range(num_objects):
        q0 = w * per_world_q + object_root_q_ids[o]
        d = perturb_xyyaw[w * num_objects + o]
        joint_q[q0] = joint_q[q0] + d[0]
        joint_q[q0 + 1] = joint_q[q0 + 1] + d[1]
        q = wp.quat(joint_q[q0 + 3], joint_q[q0 + 4], joint_q[q0 + 5], joint_q[q0 + 6])
        r = wp.normalize(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), d[2]) * q)
        joint_q[q0 + 3] = r[0]
        joint_q[q0 + 4] = r[1]
        joint_q[q0 + 5] = r[2]
        joint_q[q0 + 6] = r[3]


@dataclass
class S0Reset(ReferenceReset):
    pickup_frames: wp.array = None
    n_pickup: int = 0
    perturb_flag: wp.array = None
    perturb_xyyaw: wp.array = None
    num_objects: int = 0

    @classmethod
    def build(cls, world_count, num_frames, config=None, device=None):
        base = ReferenceReset.build(world_count, num_frames, config=config, device=device)
        ex = np.load(config.extras_path, allow_pickle=True)
        frames = np.asarray(ex["pickup_reset_frames"], np.int32) if "pickup_reset_frames" in ex.files else np.zeros(0, np.int32)
        frames = frames[(frames >= 0) & (frames < num_frames - 1)]
        num_objects = int(len(ex["object_names"]))
        self = cls(**{f: getattr(base, f) for f in base.__dataclass_fields__},
                   pickup_frames=wp.array(frames if len(frames) else np.zeros(1, np.int32), dtype=wp.int32, device=device),
                   n_pickup=int(len(frames)), perturb_flag=wp.zeros(world_count, dtype=wp.int32, device=device),
                   perturb_xyyaw=wp.zeros(world_count * num_objects, dtype=wp.vec3, device=device), num_objects=num_objects)
        _REG["policy"] = self
        print(f"[s0 reset] pickup reset frames {frames.tolist()}; perturb p={config.perturb_probability} "
              f"+-{config.perturb_position_m} m +-{config.perturb_yaw_deg} deg", flush=True)
        return self

    def sample(self, reset_mask, curriculum_voc_scale):
        c = self.config
        wp.launch(_sample, dim=self.world_count, inputs=[
            reset_mask, self.num_frames, c.voc_decay_steps, int(c.always_reset_to_first_frame), self.first_frame_probability,
            c.first_frame_voc_threshold, c.reset_finger_openness, c.seed, curriculum_voc_scale, self.pickup_frames,
            self.n_pickup, c.perturb_probability, c.perturb_position_m, math.radians(c.perturb_yaw_deg), self.num_objects],
            outputs=[self.reset_count, self.reset_frame, self.finger_scale, self.steps_since_reset, self.applied_voc_scale,
                     self.perturb_flag, self.perturb_xyyaw])

    def prepare_explicit(self, reset_mask, curriculum_voc_scale):
        super().prepare_explicit(reset_mask, curriculum_voc_scale)
        wp.launch(_clear, dim=self.world_count, inputs=[reset_mask, self.perturb_flag])


_orig_reset_worlds = _envs_base.reset_worlds


def _reset_worlds_with_perturb(model, table, state, frame_id=None, reset_frame=None, reset_mask=None,
                               finger_scale=None, reference_joint_q_offset=None):
    _orig_reset_worlds(model, table, state, frame_id=frame_id, reset_frame=reset_frame, reset_mask=reset_mask,
                       finger_scale=finger_scale, reference_joint_q_offset=reference_joint_q_offset)
    pol = _REG.get("policy")
    if pol is None or pol.config.perturb_probability <= 0.0:
        return
    mask = table.all_reset_mask if reset_mask is None else reset_mask
    wp.launch(_perturb, dim=table.world_count, inputs=[state.joint_q, table.per_world_q, mask, pol.perturb_flag,
              pol.perturb_xyyaw, table.object_root_q_ids, table.num_objects])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


_envs_base.reset_worlds = _reset_worlds_with_perturb
