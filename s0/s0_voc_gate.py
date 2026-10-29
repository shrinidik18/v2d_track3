"""Contact-gated virtual object controller (VOC) for training.

FlashCHORD's VOC pulls every object toward its reference pose (scaled by the curriculum's voc_scale) whether or not a hand
touches it, so early in training the object reward arrives without any contact. Here the assist for object o in world w is
voc_scale[w] * gate[w, o], with gate = 1 while any hand link touched o within the last `HOLD_SUBSTEPS` physics substeps,
else 0 (no assist and no gravity compensation: an untouched object just rests on the table).

Mode "wrench" (S0_VOC_GATE=wrench): gate = 1 while the S0 objective's required-wrench support score for the object exceeds
0.5 (the hands can apply the wrench the reference motion needs), and the assist carries no gravity compensation.

Installed from train_logged.py when S0_VOC_GATE is set by replacing the `apply_object_control` name that flash_chord.envs.base
calls; the live contact buffers come from the S0 objective (register()). Evaluation never uses VOC, so it is unaffected.
"""
from __future__ import annotations

import warp as wp

import flash_chord.envs.base as _base
from flash_chord.runtime.object_control import VOCParams, apply_object_control as _original
from flash_chord.utils.quat import quat_to_rotvec

import os

HOLD_SUBSTEPS = 25            # contact mode: 5 control steps at 5 substeps
WRENCH_THRESHOLD = 0.5        # wrench mode: required-wrench support score needed for assistance
WRENCH_HOLD_SUBSTEPS = 5      # wrench mode: one control step (the score is updated once per control step)
MODE = os.environ.get("S0_VOC_GATE", "contact")   # "1"/"contact" = any-contact gate, "wrench" = required-wrench gate
_REG: dict = {}


def register(contact, num_objects: int, wrench_score=None) -> None:
    cl = contact.layout
    dev = contact.contact_active.device
    W = contact.world_count
    _REG.update(contact=contact, num_objects=num_objects, world_count=W,
                hand_slot_starts=wp.array(list(cl.hand_slot_starts), dtype=wp.int32, device=dev),
                link_counts=wp.array(list(cl.link_counts), dtype=wp.int32, device=dev),
                slots_per_world=int(cl.slots_per_world), num_hands=int(cl.num_hands),
                hold=wp.zeros(W * num_objects, dtype=wp.int32, device=dev),
                gate=wp.zeros(W * num_objects, dtype=wp.float32, device=dev), wrench_score=wrench_score)


@wp.kernel
def _update_gate(contact_active: wp.array(dtype=wp.int32), slots_per_world: int, hand_slot_starts: wp.array(dtype=wp.int32),
                 link_counts: wp.array(dtype=wp.int32), num_hands: int, num_objects: int, hold_steps: int,
                 hold: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.float32)):
    i = wp.tid()
    w = i // num_objects
    o = i - w * num_objects
    touch = int(0)
    for h in range(num_hands):
        for l in range(link_counts[h]):
            if contact_active[w * slots_per_world + hand_slot_starts[h] + o * link_counts[h] + l] != 0:
                touch = 1
    if touch == 1:
        hold[i] = hold_steps
    elif hold[i] > 0:
        hold[i] = hold[i] - 1
    gate[i] = wp.where(hold[i] > 0, 1.0, 0.0)


@wp.kernel
def _update_gate_wrench(wrench_score: wp.array(dtype=wp.float32), threshold: float, hold_steps: int,
                        hold: wp.array(dtype=wp.int32), gate: wp.array(dtype=wp.float32)):
    i = wp.tid()
    if wrench_score[i] > threshold:
        hold[i] = hold_steps
    elif hold[i] > 0:
        hold[i] = hold[i] - 1
    gate[i] = wp.where(hold[i] > 0, 1.0, 0.0)


@wp.kernel
def _gated_object_control(
    body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector), body_mass: wp.array(dtype=wp.float32),
    body_com: wp.array(dtype=wp.vec3), body_inertia: wp.array(dtype=wp.mat33), object_root_body_ids: wp.array(dtype=wp.int32),
    object_body_offsets: wp.array(dtype=wp.int32), object_body_ids: wp.array(dtype=wp.int32), target_pos: wp.array(dtype=wp.vec3),
    target_quat: wp.array(dtype=wp.quat), k_lin: float, d_lin: float, k_ang: float, d_ang: float, max_force: float,
    max_torque: float, scale: wp.array(dtype=wp.float32), scale_count: int, world_count: int, object_count: int,
    bodies_per_world: int, gravity: float, mass_normalized_gains: bool, gate: wp.array(dtype=wp.float32),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    objects_per_world = object_count // world_count
    world = i // objects_per_world
    s = scale[0]
    if scale_count > 1:
        s = scale[world]
    s = s * gate[i]
    object_id = i - world * objects_per_world
    bid = object_root_body_ids[i]
    xf = body_q[bid]
    pos = wp.transform_get_translation(xf)
    quat = wp.transform_get_rotation(xf)
    vel = body_qd[bid]
    lin_vel_com = wp.spatial_top(vel)
    ang_vel = wp.spatial_bottom(vel)
    com_offset_w = wp.quat_rotate(quat, body_com[bid])
    lin_vel = lin_vel_com - wp.cross(ang_vel, com_offset_w)
    mass = float(0.0)  # noqa: UP018
    for body_index in range(object_body_offsets[object_id], object_body_offsets[object_id + 1]):
        mass += body_mass[world * bodies_per_world + object_body_ids[body_index]]
    inertia = body_inertia[bid]
    inertia_eff = (inertia[0, 0] + inertia[1, 1] + inertia[2, 2]) / 3.0
    force = k_lin * (target_pos[i] - pos) - d_lin * lin_vel
    q_err = wp.mul(target_quat[i], wp.quat_inverse(quat))
    torque = k_ang * quat_to_rotvec(q_err) - d_ang * ang_vel
    if mass_normalized_gains:
        force = mass * force
        torque = inertia_eff * torque
    force = force + wp.vec3(0.0, 0.0, gravity * mass)
    force = wp.vec3(wp.clamp(s * force[0], -max_force, max_force), wp.clamp(s * force[1], -max_force, max_force),
                    wp.clamp(s * force[2], -max_force, max_force))
    torque = wp.vec3(wp.clamp(s * torque[0], -max_torque, max_torque), wp.clamp(s * torque[1], -max_torque, max_torque),
                     wp.clamp(s * torque[2], -max_torque, max_torque))
    body_f[bid] = wp.spatial_vector(force, torque)


def gated_apply_object_control(model, state, control, command, scale, params=None):
    if not _REG or command.layout.num_articulations > 0 or command.world_count != _REG["world_count"]:
        return _original(model, state, control, command, scale, params)
    p = params or VOCParams()
    r = _REG
    n = command.world_count * command.layout.num_objects
    wrench_mode = MODE == "wrench" and r.get("wrench_score") is not None
    if wrench_mode:      # assist only while the hands' contacts can apply the motion's required wrench; no weight cancellation
        wp.launch(_update_gate_wrench, dim=n, inputs=[r["wrench_score"], WRENCH_THRESHOLD, WRENCH_HOLD_SUBSTEPS], outputs=[r["hold"], r["gate"]])
    else:
        wp.launch(_update_gate, dim=n, inputs=[r["contact"].contact_active, r["slots_per_world"], r["hand_slot_starts"], r["link_counts"],
                                               r["num_hands"], command.layout.num_objects, HOLD_SUBSTEPS], outputs=[r["hold"], r["gate"]])
    wp.launch(_gated_object_control, dim=n, inputs=[
        state.body_q, state.body_qd, model.body_mass, model.body_com, model.body_inertia, command.voc_body_ids_w,
        command.voc_object_body_offsets, command.voc_object_body_ids, command.voc_target_pos_w, command.voc_target_quat_w,
        p.k_lin, p.d_lin, p.k_ang, p.d_ang, p.max_force, p.max_torque, scale, scale.shape[0], command.world_count, n,
        command.bodies_per_world, 0.0 if wrench_mode else p.gravity, p.mass_normalized_gains, r["gate"]], outputs=[state.body_f])


def install() -> None:
    _base.apply_object_control = gated_apply_object_control
    print(f"[s0] gated VOC installed (mode={MODE})", flush=True)
