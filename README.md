# V2D Track 3 submission code: per-episode S0 policies, v1/v2 reconstruction mix

Code that produced the Track 3 submission: one FlashCHORD / FlashSAC policy per evaluation episode, trained on a
reconstructed reference of that episode (no ground truth), then rolled out with the kit's official evaluator.

Runtime: video_to_data FlashCHORD at 8c778f9 (the kit's pinned revision), Docker image
`flashchord-cuda12-575-test:193382a1-20261018`.

## Layout

| Path | What |
| --- | --- |
| `s0/` | S0 observation, action, objective, reset, actor EMA and VOC-gate code; feature builders; train / eval / queue scripts |
| `cfg/` | Hydra configs (`observation/policy_s0`, `action/relative_hand_pose`, `objective/tracking_s0`, `reset/s0_pickup`, `curriculum/tracking_96m_pickup_ray5decay`) |
| `rian_eval/` | Cleaning of the v1 references (`clean_refs.py`, `settle_spawn.py`) |
| `train_logged.py`, `monitor_training.py`, `run_in_env.sh` | Training entry point, checkpoint monitor, container runner |
| `selection.txt` | Which reference version and checkpoint each episode uses |
| `checkpoints_v1_container_paths.json`, `checkpoints_v2_container_paths.json` | Checkpoints passed to the official evaluator |

## References

- **v1:** Ego2Sim reconstruction v1 (2026-10-21), cleaned with `rian_eval/` (cup spin removed, 180° flips unwrapped,
  spawn poses settled). Features: `s0/s0_features_eval.py`. Runs: `s0/start_eval_queue.sh`.
- **v2:** Ego2Sim reconstruction v2 (2026-10-27), used as delivered. Features: `s0/s0_features_v2.py`. Runs:
  `s0/start_v2_queue.sh` (ep1, ep15), `s0/start_v2_all_queue.sh` (the other 18).

Both use the same recipe: 96.8M environment steps, `tracking_96m_pickup_ray5decay` curriculum, pickup and
object-perturbed resets, actor weight EMA. Each run's checkpoint is the best by ADD-AUC over 64 parallel rollouts.

## Per-episode selection

For each episode, the v2 policy is used when its best-checkpoint 64-world ADD-AUC beats the v1 policy's by at least
0.01, otherwise the v1 policy (`selection.txt`). Meshes come from the same reconstruction version as the policy.
Both scores are measured against that version's own reference, not ground truth.
