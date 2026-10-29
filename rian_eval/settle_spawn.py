"""Frame-0 spawn settle for a reference parquet: objects start at frame 0, the object controller (replay gains) pulls them toward
that pose, contacts with the table / support and each other are on, hands do not collide, gravity on. After 1.5 s the
objects sit at the closest non-penetrating pose; written as JSON (per object: position, quat xyzw).
    python settle_spawn.py <parquet_dir> <out.json>
"""
import json
import sys

import numpy as np
import warp as wp
from hydra import compose, initialize_config_dir

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene

parquet, out = sys.argv[1], sys.argv[2]
with initialize_config_dir(config_dir="/fc/src/flash_chord/configs", version_base="1.3"):
    cfg = compose(config_name="replay", overrides=[f"task.parquet='{parquet}'", "task.motion_speed=1.0",
                                                   "collision.robot_object_collision=false"] + sys.argv[3:])
rc = instantiate_typed(cfg.replay, ReplayConfig)
setup = setup_scene(parquet=cfg.task.parquet, control_fps=rc.sim.fps, motion_speed=1.0,
                    embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
                    world_count=1, include_support=True, decompose_objects=True)
runner = ReplayRunner(setup.scene, setup.reference, rc, device="cuda:0")
runner.reset(0)
ids = runner.command.body_ids_w.numpy().ravel()
q0 = runner.state_0.body_q.numpy()[ids].copy()
for _ in range(int(1.5 * rc.sim.fps)):
    runner.step(0)
q = runner.state_0.body_q.numpy()[ids]
names = list(setup.reference.object_body_names())
res = {"objects": names, "position": q[:, :3].tolist(), "quat_xyzw": q[:, 3:7].tolist(),
       "start_position": q0[:, :3].tolist(), "start_quat_xyzw": q0[:, 3:7].tolist(), "moved_cm": (100 * np.linalg.norm(q[:, :3] - q0[:, :3], axis=1)).tolist()}
json.dump(res, open(out, "w"), indent=1)
print("settle:", {n: round(m, 2) for n, m in zip(names, res["moved_cm"])})
