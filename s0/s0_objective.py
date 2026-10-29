"""FlashCHORD's tracking objective plus a `hand_ray` term: hand guidance from the egocentric video that needs no depth.

hand_ray (per world, in [0, 1]) = mean over valid keypoint rays (wrist + 5 fingertips per hand, current frame) of
exp(-d / ray_scale), d = distance from the robot's matching point (palm / fingertip frame) to the ray through the 2D
keypoint. Its weight comes from the curriculum like every other term (name `hand_ray`). Select with `objective=tracking_s0`.

object_contact (per world, in [0, 1]): for every object whose reference trajectory moves within the next
`contact_lookahead` frames (> 1 cm or > 5 deg), 0.5 * exp(-d / contact_scale) + 0.5 * [a hand link touches it], with d the
closest fingertip-to-surface distance (96 object-frame surface points); averaged over those objects, 0 when none moves.
Built only from the object trajectory and live simulator contacts, never from the reconstructed hand.

required_wrench (per world, in [0, 1]): for every object that needs a wrench at this frame (lifted/tilted off its rest pose,
or moving on the table), the wrench its reference motion requires is estimated from the object trajectory:
force m*(a_ref + g*z) when lifted (table friction mu_t*m*g along the motion when sliding on the table), torque I*alpha_ref
(plus friction along the rotation on the table). The combined friction-cone support of both hands' live contacts is evaluated
along that 6-D direction and along 8 directions on a cone around it (20 deg on the table, 60 deg when lifted, i.e. margin
against dropping); score = min over the cone / 0.5, clamped to [0, 1], averaged over objects that need a wrench.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from flash_chord.objectives.config import ObjectiveConfig

RAY_SIDES = ("left", "right")


@dataclass(frozen=True)
class S0ObjectiveConfig:
    base: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    extras_path: str = ""
    ray_scale: float = 0.05
    contact_scale: float = 0.03
    contact_lookahead: int = 10
    wrench_friction: float = 0.8          # friction-cone coefficient for the support (sim contact friction is 1.0)
    wrench_cone_table_deg: float = 20.0
    wrench_cone_lifted_deg: float = 60.0
    wrench_table_friction: float = 0.5    # table friction used for the push direction

    def build(self, model, embodiment, robot_reference, reference, command, contact, action, world_count, frame_dt,
              device=None, reference_joint_q_offset=None, episode_start_frame=None):
        inner = self.base.build(model, embodiment, robot_reference, reference, command, contact, action, world_count, frame_dt,
                                device=device, reference_joint_q_offset=reference_joint_q_offset,
                                episode_start_frame=episode_start_frame)
        return S0Objective.build(self, inner, model, embodiment, world_count, frame_dt, device, reference=reference,
                                 command=command, contact=contact)


@wp.kernel
def _compose_s0(inner_terms: wp.array(dtype=wp.float32), inner_score: wp.array(dtype=wp.float32), n_inner: int,
                body_q: wp.array(dtype=wp.transform), bodies_per_world: int, timestep: wp.array(dtype=wp.int32), num_frames: int,
                ray_origin: wp.array(dtype=wp.vec3), ray_dir: wp.array(dtype=wp.vec3), ray_valid: wp.array(dtype=wp.float32),
                ray_body_ids: wp.array(dtype=wp.int32), ray_body_pos: wp.array(dtype=wp.vec3), ray_points: int,
                ray_scale: float, ray_weight: wp.array(dtype=wp.float32), frame_dt: float,
                num_objects: int, object_body_ids: wp.array(dtype=wp.int32), moving: wp.array(dtype=wp.float32),
                surf: wp.array(dtype=wp.vec3), n_surf: int, tip_body_ids: wp.array(dtype=wp.int32), tip_body_pos: wp.array(dtype=wp.vec3),
                n_tips: int, contact_active: wp.array(dtype=wp.int32), slots_per_world: int, hand_slot_starts: wp.array(dtype=wp.int32),
                link_counts: wp.array(dtype=wp.int32), num_hands: int, contact_scale: float, contact_weight: wp.array(dtype=wp.float32),
                contact_pos_o: wp.array(dtype=wp.vec3), contact_dir_o: wp.array(dtype=wp.vec3), obj_rc: wp.array(dtype=wp.float32),
                wrench_active: wp.array(dtype=wp.float32), wrench_dirs: wp.array(dtype=wp.spatial_vector), n_dirs: int, mu: float,
                wrench_weight: wp.array(dtype=wp.float32), wrench_score: wp.array(dtype=wp.float32),
                terms: wp.array(dtype=wp.float32), score: wp.array(dtype=wp.float32)):
    w = wp.tid()
    n = n_inner + 3
    for i in range(n_inner):
        terms[w * n + i] = inner_terms[w * n_inner + i]
    frame = wp.clamp(timestep[w], 0, num_frames - 1)
    o = ray_origin[frame]
    acc = float(0.0)
    cnt = float(0.0)
    for j in range(ray_points):
        if ray_valid[frame * ray_points + j] > 0.5:
            p = wp.transform_point(body_q[w * bodies_per_world + ray_body_ids[j]], ray_body_pos[j])
            d = ray_dir[frame * ray_points + j]
            s = wp.max(wp.dot(p - o, d), 0.0)
            acc += wp.exp(-wp.length(o + d * s - p) / ray_scale)
            cnt += 1.0
    r = float(0.0)
    if cnt > 0.0:
        r = acc / cnt
    terms[w * n + n_inner] = r
    cacc = float(0.0)
    ccnt = float(0.0)
    for ob in range(num_objects):
        if moving[frame * num_objects + ob] > 0.5:
            xf = body_q[w * bodies_per_world + object_body_ids[ob]]
            dmin = float(1.0e6)
            for t in range(n_tips):
                tp = wp.transform_point(body_q[w * bodies_per_world + tip_body_ids[t]], tip_body_pos[t])
                for k in range(n_surf):
                    dmin = wp.min(dmin, wp.length(wp.transform_point(xf, surf[ob * n_surf + k]) - tp))
            touch = float(0.0)
            for h in range(num_hands):
                for l in range(link_counts[h]):
                    if contact_active[w * slots_per_world + hand_slot_starts[h] + ob * link_counts[h] + l] != 0:
                        touch = 1.0
            cacc += 0.5 * wp.exp(-dmin / contact_scale) + 0.5 * touch
            ccnt += 1.0
    rc = float(0.0)
    if ccnt > 0.0:
        rc = cacc / ccnt
    terms[w * n + n_inner + 1] = rc
    wacc = float(0.0)
    wcnt = float(0.0)
    for ob in range(num_objects):
        sc = float(0.0)
        if wrench_active[frame * num_objects + ob] > 0.5:
            smin = float(1.0e6)
            for j in range(n_dirs):
                dvec = wrench_dirs[(frame * num_objects + ob) * n_dirs + j]
                best = float(0.0)
                for h in range(num_hands):
                    for l in range(link_counts[h]):
                        slot = w * slots_per_world + hand_slot_starts[h] + ob * link_counts[h] + l
                        if contact_active[slot] != 0:
                            pc = contact_pos_o[slot]
                            nc = wp.normalize(contact_dir_o[slot])
                            t1 = wp.cross(nc, wp.vec3(1.0, 0.0, 0.0))
                            if wp.length(t1) < 0.1:
                                t1 = wp.cross(nc, wp.vec3(0.0, 1.0, 0.0))
                            t1 = wp.normalize(t1)
                            t2 = wp.cross(nc, t1)
                            for e in range(9):
                                f = nc
                                if e < 8:
                                    ang = float(e) * 0.7853981634
                                    f = wp.normalize(nc + mu * (wp.cos(ang) * t1 + wp.sin(ang) * t2))
                                tq = wp.cross(pc, f) / obj_rc[ob]
                                val = wp.dot(wp.spatial_top(dvec), f) + wp.dot(wp.spatial_bottom(dvec), tq)
                                best = wp.max(best, val)
                smin = wp.min(smin, best)
            sc = wp.clamp(smin / 0.5, 0.0, 1.0)
            wacc += sc
            wcnt += 1.0
        wrench_score[w * num_objects + ob] = sc
    rw = float(0.0)
    if wcnt > 0.0:
        rw = wacc / wcnt
    terms[w * n + n_inner + 2] = rw
    score[w] = inner_score[w] + frame_dt * (ray_weight[0] * r + contact_weight[0] * rc + wrench_weight[0] * rw)


@dataclass
class S0Objective:
    inner: object
    config: S0ObjectiveConfig
    world_count: int
    bodies_per_world: int
    frame_dt: float
    num_frames: int
    arrays: dict
    term_names: tuple
    terms: wp.array
    weights: wp.array
    score: wp.array
    ray_weight: wp.array
    contact_weight: wp.array = None
    command: object = None
    contact: object = None
    wrench_weight: wp.array = None
    wrench_score: wp.array = None

    @classmethod
    def build(cls, config, inner, model, embodiment, world_count, frame_dt, device, reference=None, command=None, contact=None):
        ex = np.load(config.extras_path, allow_pickle=True)
        T = int(ex["ray_origin"].shape[0])
        ids, pos = [], []
        for side in RAY_SIDES:
            hand = embodiment.hand(side)
            for fr in (hand.palm_frame, *hand.fingertip_frames):
                ids.append(fr.body_id); pos.append(fr.body_to_frame_pos)
        f32 = lambda a, dt=wp.float32: wp.array(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        arrays = dict(ray_origin=f32(ex["ray_origin"].astype(np.float32), wp.vec3),
                      ray_dir=f32(ex["ray_dir"].astype(np.float32).reshape(-1, 3), wp.vec3),
                      ray_valid=f32(ex["ray_valid"].astype(np.float32).reshape(-1)),
                      ray_body_ids=wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=device),
                      ray_body_pos=f32(np.asarray(pos, np.float32), wp.vec3), ray_points=len(ids))
        # object-motion flags from the reference object trajectory (frame f: the object moves within [f, f + lookahead])
        from scipy.spatial.transform import Rotation as Rot
        P = np.asarray(reference.object_body_pos_w(), np.float64); Q = np.asarray(reference.object_body_quat_w(), np.float64)  # (T,B,3) (T,B,4 wxyz)
        Tn, B = P.shape[0], P.shape[1]
        moving = np.zeros((Tn, B), np.float32)
        for f in range(Tn):
            g = min(f + config.contact_lookahead, Tn - 1)
            for b in range(B):
                dp = np.linalg.norm(P[f:g + 1, b] - P[f, b], axis=-1).max()
                r0 = Rot.from_quat(Q[f, b, [1, 2, 3, 0]])
                da = np.degrees((Rot.from_quat(Q[f:g + 1, b][:, [1, 2, 3, 0]]) * r0.inv()).magnitude()).max()
                moving[f, b] = float(dp > 0.01 or da > 5.0)
        tip_ids, tip_pos = [], []
        for side in RAY_SIDES:
            for fr in embodiment.hand(side).fingertip_frames:
                tip_ids.append(fr.body_id); tip_pos.append(fr.body_to_frame_pos)
        cl = contact.layout
        arrays.update(moving=f32(moving.reshape(-1)), surf=f32(ex["object_surface_points"].astype(np.float32).reshape(-1, 3), wp.vec3),
                      n_surf=int(ex["object_surface_points"].shape[1]), num_objects=B,
                      tip_body_ids=wp.array(np.asarray(tip_ids, np.int32), dtype=wp.int32, device=device),
                      tip_body_pos=f32(np.asarray(tip_pos, np.float32), wp.vec3), n_tips=len(tip_ids),
                      hand_slot_starts=wp.array(np.asarray(cl.hand_slot_starts, np.int32), dtype=wp.int32, device=device),
                      link_counts=wp.array(np.asarray(cl.link_counts, np.int32), dtype=wp.int32, device=device),
                      slots_per_world=int(cl.slots_per_world), num_hands=int(cl.num_hands))
        print(f"[s0 objective] object moving fraction per object: {moving.mean(0).round(2).tolist()}", flush=True)
        active, dirs = required_wrench_directions(P, Q, np.asarray(reference.object_mesh_radius(), np.float64), float(reference.fps), config)
        print(f"[s0 objective] required-wrench active fraction per object: {active.mean(0).round(2).tolist()}", flush=True)
        arrays.update(wrench_active=f32(active.reshape(-1)), wrench_dirs=wp.array(dirs.reshape(-1, 6).astype(np.float32), dtype=wp.spatial_vector, device=device),
                      n_dirs=int(dirs.shape[2]), rc=f32(np.asarray(reference.object_mesh_radius(), np.float32)))
        wrench_score = wp.zeros(world_count * B, dtype=wp.float32, device=device)
        try:                                      # expose live contacts / wrench support to the gated VOC (training process only)
            import s0_voc_gate
            s0_voc_gate.register(contact, B, wrench_score=wrench_score)
        except ImportError:
            pass
        names = tuple(inner.term_names) + ("hand_ray", "object_contact", "required_wrench")
        return cls(inner=inner, config=config, world_count=world_count, bodies_per_world=model.body_count // world_count,
                   frame_dt=float(frame_dt), num_frames=T, arrays=arrays, term_names=names,
                   terms=wp.zeros(world_count * len(names), dtype=wp.float32, device=device),
                   weights=wp.zeros(len(names), dtype=wp.float32, device=device),
                   score=wp.zeros(world_count, dtype=wp.float32, device=device),
                   ray_weight=wp.zeros(1, dtype=wp.float32, device=device),
                   contact_weight=wp.zeros(1, dtype=wp.float32, device=device), command=command, contact=contact,
                   wrench_weight=wp.zeros(1, dtype=wp.float32, device=device), wrench_score=wrench_score)

    def evaluate(self, state, timestep, terminated) -> None:
        self.inner.evaluate(state, timestep, terminated)
        a = self.arrays
        wp.launch(_compose_s0, dim=self.world_count, inputs=[
            self.inner.terms, self.inner.score, len(self.inner.term_names), state.body_q, self.bodies_per_world, timestep,
            self.num_frames, a["ray_origin"], a["ray_dir"], a["ray_valid"], a["ray_body_ids"], a["ray_body_pos"],
            a["ray_points"], self.config.ray_scale, self.ray_weight, self.frame_dt,
            a["num_objects"], self.command.body_ids, a["moving"], a["surf"], a["n_surf"], a["tip_body_ids"], a["tip_body_pos"],
            a["n_tips"], self.contact.contact_active, a["slots_per_world"], a["hand_slot_starts"], a["link_counts"], a["num_hands"],
            self.config.contact_scale, self.contact_weight, self.contact.contact_pos_o, self.contact.contact_force_direction_o,
            a["rc"], a["wrench_active"], a["wrench_dirs"], a["n_dirs"], self.config.wrench_friction, self.wrench_weight,
            self.wrench_score], outputs=[self.terms, self.score])

    def reset(self, reset_mask) -> None:
        self.inner.reset(reset_mask)

    def set_weights(self, weights) -> None:
        extra = ("hand_ray", "object_contact", "required_wrench")
        inner_w = {k: v for k, v in weights.items() if k not in extra}
        self.inner.set_weights(inner_w)                      # validates the FlashCHORD terms exactly
        full = {**{k: 0.0 for k in extra}, **{k: float(v) for k, v in weights.items()}}   # S0 terms missing in older configs -> 0
        self.ray_weight.assign(np.asarray([full["hand_ray"]], np.float32))
        self.contact_weight.assign(np.asarray([full["object_contact"]], np.float32))
        self.wrench_weight.assign(np.asarray([full["required_wrench"]], np.float32))
        self.weights.assign(np.asarray([full[n] for n in self.term_names], np.float32))


def required_wrench_directions(P, Q, rc, fps, cfg, n_perturb=8, seed=0):
    """Per frame and object: whether the reference motion needs a wrench, and 1 + n_perturb unit 6-D directions
    (force; torque / rc) - the required wrench and a cone around it. P (T,B,3), Q (T,B,4 wxyz), rc (B,)."""
    from scipy.spatial.transform import Rotation as Rot
    T, B = P.shape[:2]
    L, dt, g, m = cfg.contact_lookahead, 1.0 / fps, 9.81, 0.3
    active = np.zeros((T, B), np.float32); dirs = np.zeros((T, B, 1 + n_perturb, 6), np.float32)
    rng = np.random.default_rng(seed)
    for b in range(B):
        rots = Rot.from_quat(Q[:, b][:, [1, 2, 3, 0]]); z0 = P[0, b, 2]
        up_obj = rots[0].inv().apply([0.0, 0.0, 1.0])          # object axis that was vertical at rest
        I = 0.4 * m * rc[b] ** 2
        for f in range(T):
            fp, fm = min(f + L, T - 1), max(f - L, 0)
            tilt = np.degrees(np.arccos(np.clip(rots[f].apply(up_obj)[2], -1, 1)))
            lifted = (P[f, b, 2] - z0 > 0.02) or tilt > 15.0
            disp = np.linalg.norm(P[fp, b] - P[f, b]); rot_ahead = np.degrees((rots[fp] * rots[f].inv()).magnitude())
            moving = disp > 0.01 or rot_ahead > 5.0
            if not (lifted or moving):
                continue
            v = (P[fp, b] - P[f, b]) / max((fp - f) * dt, dt)
            a = (P[fp, b] - 2 * P[f, b] + P[fm, b]) / max(((fp - f) * dt) * ((f - fm) * dt), dt * dt)
            w_f = (rots[fp] * rots[f].inv()).as_rotvec() / max((fp - f) * dt, dt)
            w_b = (rots[f] * rots[fm].inv()).as_rotvec() / max((f - fm) * dt, dt)
            alpha = (w_f - w_b) / max(L * dt, dt)
            force = m * a
            if lifted:
                force = force + np.array([0.0, 0.0, m * g])
            elif np.linalg.norm(v) > 0.02:
                force = force + cfg.wrench_table_friction * m * g * v / np.linalg.norm(v)
            torque = I * alpha
            if not lifted and np.linalg.norm(w_f) > 0.1:
                torque = torque + 0.1 * m * g * rc[b] * w_f / np.linalg.norm(w_f)
            d0 = np.concatenate([force, torque / rc[b]])
            if np.linalg.norm(d0) < 1e-6:
                continue
            d0 /= np.linalg.norm(d0)
            th = np.radians(cfg.wrench_cone_lifted_deg if lifted else cfg.wrench_cone_table_deg)
            out = [d0]
            for _ in range(n_perturb):
                u = rng.normal(size=6); u -= (u @ d0) * d0; u /= np.linalg.norm(u)
                out.append(np.cos(th) * d0 + np.sin(th) * u)
            dirs[f, b] = np.asarray(out, np.float32); active[f, b] = 1.0
    return active, dirs
