"""S0 policy observation: FlashCHORD's policy observation plus future object trajectory, object shape code, hand rays from
the egocentric video, and spatial image features of the reference frame. Selected from Hydra with `observation=policy_s0`
(config in /env/cfg/observation/policy_s0.yaml); the FlashCHORD source is not modified.

Appended blocks, per world (frame = current reference frame, future frames clamped to the last one):
  object_future_relative_pose   for each offset k and object: reference pose at frame+k in the current object frame
                                (position 3 + wxyz 4)
  object_shape_bps32            32-d basis-point-set shape code per object (constant within an episode)
  hand_ray_features             for each ray offset r, hand (left, right) and keypoint (wrist, thumb..pinky tips):
                                offset from the robot's matching point to the nearest point on the ray (3), ray direction
                                (3), valid (1)
  ego_image_features            spatial-softmax keypoints + hand-crop features of the reference video frame
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from flash_chord.envs.observation import PolicyObservation, PolicyObservationConfig, unique_quat
from flash_chord.utils.quat import quat_inv_xyzw, quat_mul_xyzw

RAY_SIDES = ("left", "right")          # order of the hand axis in the extras file
RAY_POINTS = 6                          # wrist + 5 fingertips


@dataclass(frozen=True)
class S0ObservationConfig:
    base: PolicyObservationConfig = field(default_factory=PolicyObservationConfig)
    extras_path: str = ""
    trajectory_offsets: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    ray_offsets: tuple[int, ...] = (0, 5, 10)
    include_shape: bool = True
    include_image: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_offsets", tuple(int(k) for k in self.trajectory_offsets))
        object.__setattr__(self, "ray_offsets", tuple(int(k) for k in self.ray_offsets))
        if not self.extras_path:
            raise ValueError("observation.extras_path must point at the episode's S0 extras npz")
        if any(k < 0 for k in self.trajectory_offsets + self.ray_offsets):
            raise ValueError("offsets must be non-negative")

    def build(self, scene, action, command, contact, timestep, episode_step, reference_joint_q_offset, device=None):
        return S0Observation.build(self, scene, action, command, contact, timestep, episode_step, device=device)


@wp.kernel
def _append_s0(
    inner: wp.array(dtype=wp.float32),
    inner_dim: int,
    out_dim: int,
    body_q: wp.array(dtype=wp.transform),
    bodies_per_world: int,
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    num_objects: int,
    object_body_ids: wp.array(dtype=wp.int32),
    ref_pos: wp.array(dtype=wp.vec3),
    ref_quat: wp.array(dtype=wp.quat),
    traj_offsets: wp.array(dtype=wp.int32),
    bps: wp.array(dtype=wp.float32),
    bps_dim: int,
    ray_origin: wp.array(dtype=wp.vec3),
    ray_dir: wp.array(dtype=wp.vec3),
    ray_valid: wp.array(dtype=wp.float32),
    ray_offsets: wp.array(dtype=wp.int32),
    ray_body_ids: wp.array(dtype=wp.int32),
    ray_body_pos: wp.array(dtype=wp.vec3),
    ray_points: int,
    image: wp.array(dtype=wp.float32),
    image_dim: int,
    out: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    src = w * inner_dim
    dst = w * out_dim
    for i in range(inner_dim):
        out[dst + i] = inner[src + i]
    c = dst + inner_dim
    frame = wp.clamp(timestep[w], 0, num_frames - 1)
    base = w * bodies_per_world
    for k in range(traj_offsets.shape[0]):
        f2 = wp.min(frame + traj_offsets[k], num_frames - 1)
        for o in range(num_objects):
            xf = body_q[base + object_body_ids[o]]
            rel_p = wp.transform_point(wp.transform_inverse(xf), ref_pos[f2 * num_objects + o])
            rel_q = quat_mul_xyzw(quat_inv_xyzw(wp.transform_get_rotation(xf)), unique_quat(ref_quat[f2 * num_objects + o]))
            out[c + 0] = rel_p[0]
            out[c + 1] = rel_p[1]
            out[c + 2] = rel_p[2]
            out[c + 3] = rel_q[3]
            out[c + 4] = rel_q[0]
            out[c + 5] = rel_q[1]
            out[c + 6] = rel_q[2]
            c += 7
    for i in range(bps_dim):
        out[c + i] = bps[i]
    c += bps_dim
    for r in range(ray_offsets.shape[0]):
        f2 = wp.min(frame + ray_offsets[r], num_frames - 1)
        origin = ray_origin[f2]
        for j in range(ray_points):
            p = wp.transform_point(body_q[base + ray_body_ids[j]], ray_body_pos[j])
            d = ray_dir[f2 * ray_points + j]
            s = wp.max(wp.dot(p - origin, d), 0.0)
            off = origin + d * s - p
            out[c + 0] = off[0]
            out[c + 1] = off[1]
            out[c + 2] = off[2]
            out[c + 3] = d[0]
            out[c + 4] = d[1]
            out[c + 5] = d[2]
            out[c + 6] = ray_valid[f2 * ray_points + j]
            c += 7
    for i in range(image_dim):
        out[c + i] = image[frame * image_dim + i]


@dataclass
class S0Observation:
    inner: PolicyObservation
    config: S0ObservationConfig
    world_count: int
    bodies_per_world: int
    timestep: wp.array
    command: object
    num_frames: int
    num_objects: int
    arrays: dict
    observation_dim: int
    block_names: tuple
    block_ranges: tuple
    observation: wp.array

    @classmethod
    def build(cls, config, scene, action, command, contact, timestep, episode_step, device=None):
        inner = PolicyObservation.build(scene, action, command, contact, timestep, episode_step, config=config.base, device=device)
        ex = np.load(config.extras_path, allow_pickle=True)
        T = int(ex["ray_origin"].shape[0])
        if command.reference.num_frames != T:
            raise ValueError(f"extras have {T} frames; the reference has {command.reference.num_frames}")
        num_objects = command.layout.num_bodies
        bps = ex["bps"].astype(np.float32).reshape(-1) if config.include_shape else np.zeros(0, np.float32)
        if config.include_shape and len(bps) != 32 * num_objects:
            raise ValueError(f"bps code has {len(bps)} values for {num_objects} objects")
        image = ex["image_features"].astype(np.float32) if config.include_image else np.zeros((T, 0), np.float32)
        body_ids, body_pos = [], []
        for side in RAY_SIDES:
            hand = scene.layout.hand(side)
            for fr in (hand.palm_frame, *hand.fingertip_frames):
                body_ids.append(fr.body_id); body_pos.append(fr.body_to_frame_pos)
        ray_points = len(RAY_SIDES) * RAY_POINTS
        f32 = lambda a, dt=wp.float32: wp.array(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        arrays = {
            "traj_offsets": wp.array(np.asarray(config.trajectory_offsets, np.int32), dtype=wp.int32, device=device),
            "bps": f32(bps),
            "ray_origin": f32(ex["ray_origin"].astype(np.float32), wp.vec3),
            "ray_dir": f32(ex["ray_dir"].astype(np.float32).reshape(-1, 3), wp.vec3),
            "ray_valid": f32(ex["ray_valid"].astype(np.float32).reshape(-1)),
            "ray_offsets": wp.array(np.asarray(config.ray_offsets, np.int32), dtype=wp.int32, device=device),
            "ray_body_ids": wp.array(np.asarray(body_ids, np.int32), dtype=wp.int32, device=device),
            "ray_body_pos": f32(np.asarray(body_pos, np.float32), wp.vec3),
            "image": f32(image.reshape(-1)),
            "ray_points": ray_points,
            "image_dim": int(image.shape[1]),
        }
        names, ranges = list(inner.block_names), list(inner.block_ranges)
        cursor = inner.observation_dim

        def add(name, width):
            nonlocal cursor
            if width:
                names.append(name); ranges.append((cursor, cursor + width)); cursor += width
        add(f"object_future_relative_pose_x{len(config.trajectory_offsets)}", 7 * num_objects * len(config.trajectory_offsets))
        add("object_shape_bps32", len(bps))
        add(f"hand_ray_offset_dir_valid_x{len(config.ray_offsets)}", 7 * ray_points * len(config.ray_offsets))
        add(f"ego_image_features_{image.shape[1]}", image.shape[1])
        world_count = scene.world_count
        return cls(inner=inner, config=config, world_count=world_count,
                   bodies_per_world=scene.model.body_count // world_count, timestep=timestep, command=command,
                   num_frames=T, num_objects=num_objects, arrays=arrays, observation_dim=cursor,
                   block_names=tuple(names), block_ranges=tuple(ranges),
                   observation=wp.zeros(world_count * cursor, dtype=wp.float32, device=device))

    def compute(self, state) -> wp.array:
        return self.compute_into(state, self.observation)

    def compute_into(self, state, output: wp.array) -> wp.array:
        inner = self.inner.compute(state)
        a = self.arrays
        wp.launch(_append_s0, dim=self.world_count, inputs=[
            inner, self.inner.observation_dim, self.observation_dim, state.body_q, self.bodies_per_world, self.timestep,
            self.num_frames, self.num_objects, self.command.body_ids, self.command.reference.body_pos_w,
            self.command.reference.body_quat_w, a["traj_offsets"], a["bps"], a["bps"].shape[0],
            a["ray_origin"], a["ray_dir"], a["ray_valid"], a["ray_offsets"], a["ray_body_ids"], a["ray_body_pos"],
            a["ray_points"], a["image"], a["image_dim"]], outputs=[output])
        return output
