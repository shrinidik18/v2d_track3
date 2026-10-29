"""S0 vs baseline (same episode, same recipe, teammate hands): per-checkpoint 64-world evaluation curves and final table."""
import glob, json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req")
RUNS = {"baseline": ("training/ep{e}_teammate", "training/ep{e}_ft_objonly"), "S0": ("training_s0/ep{e}_s0_p1_80M", "training_s0/ep{e}_s0_p2_ft_objonly")}
KEYS = [("add_auc", "ADD AUC"), ("object_position_error_m", "object position error (m)"),
        ("object_orientation_error_deg", "object orientation error (deg)"), ("mppe_cm", "MPPE (cm)")]
lines = []
fig, ax = plt.subplots(2, 4, figsize=(20, 8))
for row, e in enumerate((40, 41)):
    for label, dirs in RUNS.items():
        pts = []
        for d in dirs:
            for f in glob.glob(str(R / d.format(e=e) / "eval/metrics_*.json")):
                m = json.loads(Path(f).read_text())["metrics"]; pts.append((int(re.findall(r"metrics_(\d+)", f)[0]), m))
        pts.sort()
        for col, (k, title) in enumerate(KEYS):
            if pts: ax[row, col].plot([s / 1e6 for s, _ in pts], [m[k] for _, m in pts], marker="o", ms=3, label=label)
            ax[row, col].set_title(f"ep{e}: {title}"); ax[row, col].set_xlabel("env steps (M)"); ax[row, col].grid(alpha=.3)
        for wc in ("w1", "w64"):
            f = R / dirs[1].format(e=e) / f"eval_final/metrics_96776192_{wc}.json"
            if f.exists():
                m = json.loads(f.read_text())["metrics"]
                lines.append(f"ep{e} {label:8s} {wc:3s}: AUC {m['add_auc']:.4f}  MPPE {m['mppe_cm']:.2f} cm  pos {100*m['object_position_error_m']:.2f} cm  "
                             f"rot {m['object_orientation_error_deg']:.2f} deg  SP-SR {m['spider_sr_uncentered']:.2f}  MP-SR {m['maniptrans_sr']:.2f}")
    ax[row, 0].legend()
fig.suptitle("S0 (rays + image + shape + future trajectory, no reference-hand inputs) vs baseline, 64-world eval per checkpoint")
fig.tight_layout(); fig.savefig(R / "training_s0/s0_vs_baseline.png", dpi=110)
(R / "training_s0/s0_vs_baseline.txt").write_text("final checkpoint (96.8M), official evaluator\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
