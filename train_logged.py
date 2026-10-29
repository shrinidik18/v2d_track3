"""Run FlashCHORD's train_flash_sac.py with its W&B logger replaced by a local JSONL writer (no network, no account).

Every logged metric dict becomes one line {"step": N, ...} in $METRICS_JSONL. The FlashCHORD source is untouched.
"""
import json
import os
import runpy
import sys
import types

OUT = os.environ.get("METRICS_JSONL", "/run_out/metrics.jsonl")


class _Run:
    id = "local"

    def log(self, metrics, step=None, **_):
        row = {"step": step, **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}}
        with open(OUT, "a") as f:
            f.write(json.dumps(row) + "\n")

    def log_artifact(self, *a, **k): pass
    def finish(self, *a, **k): pass


class _Artifact:
    def __init__(self, *a, **k): pass
    def add_file(self, *a, **k): pass


wandb = types.ModuleType("wandb")
wandb.init = lambda **kw: _Run()
wandb.Artifact = _Artifact
sys.modules["wandb"] = wandb

_thr = os.environ.get("COACD_THRESHOLD")
if _thr:   # finer collision hulls (FlashCHORD's default is threshold 0.05); patched here, the source stays untouched
    import flash_chord.assets.mesh as _mesh
    _orig = _mesh.convex_decompose
    def _patched(vertices, faces, **kw):
        kw["threshold"] = float(_thr)
        return _orig(vertices, faces, **kw)
    _mesh.convex_decompose = _patched

if os.environ.get("S0_VOC_GATE"):   # contact-gated virtual object controller (s0/s0_voc_gate.py)
    import s0_voc_gate
    s0_voc_gate.install()

if os.environ.get("S0_EMA"):        # averaged actor weights in the policy checkpoints (s0/s0_ema.py)
    import s0_ema
    s0_ema.install(float(os.environ["S0_EMA"]))

script = "/fc/scripts/train_flash_sac.py"
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
