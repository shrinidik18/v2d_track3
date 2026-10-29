"""All S0 experiments of one episode (training_s0/ep<n>_v*/) plus the old-recipe baseline: per-checkpoint 64-world curves
and the final official-evaluator table.   python s0_compare_episode.py <episode>   -> training_s0/_summaries/ep<n>_compare.*"""
import glob, json, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req"); S = R / "training_s0"
ep = int(sys.argv[1])
BASE = {18: ("training/ep18_ours_p1_80M", "training/ep18_ours_p2_ft_objonly"), 40: ("training/ep40_teammate", "training/ep40_ft_objonly"),
        41: ("training/ep41_teammate", "training/ep41_ft_objonly"), 11: ("training/ep11_teammate_p1_80M", "training/ep11_teammate_p2_ft_objonly")}
runs = {}
if ep in BASE: runs[f"ep{ep}_v0_baseline (old recipe)"] = BASE[ep]
for d in sorted(S.glob(f"ep{ep}_v*"), key=lambda p: int(re.search(r"_v(\d+)", p.name).group(1))):
    phases = sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.startswith(("p1_", "p2_")))
    runs[d.name] = tuple(str(d.relative_to(R) / p) for p in phases) if len(phases) == 2 else (str(d.relative_to(R) / phases[0]),) * 2
KEYS = [("add_auc", "ADD AUC"), ("object_position_error_m", "object position error (m)"), ("object_orientation_error_deg", "object orientation error (deg)"), ("mppe_cm", "MPPE (cm)")]
fig, ax = plt.subplots(1, 4, figsize=(26, 6)); lines = []
for label, dirs in runs.items():
    pts = sorted({int(re.findall(r"metrics_(\d+)", f)[0]): json.loads(Path(f).read_text())["metrics"] for d in set(dirs) for f in glob.glob(str(R / d / "eval/metrics_*.json"))}.items())
    for col, (k, title) in enumerate(KEYS):
        if pts: ax[col].plot([s / 1e6 for s, _ in pts], [m[k] for _, m in pts], marker="o", ms=3, label=label)
        ax[col].set_title(f"ep{ep}: {title}"); ax[col].set_xlabel("env steps (M)"); ax[col].grid(alpha=.3)
    if pts:
        best = max(pts, key=lambda x: x[1]["add_auc"]); lines.append(f"{label:48s} best 64-world checkpoint: {best[0]/1e6:.1f}M AUC {best[1]['add_auc']:.3f}")
    for wc in ("w1", "w64"):
        f = next(iter(sorted(glob.glob(str(R / dirs[1] / f"eval_final/metrics_96776192_{wc}.json")) + sorted(glob.glob(str(R / dirs[1] / f"eval_final/metrics_79998976_{wc}.json"))))), None)
        f = Path(f) if f else R / "missing"
        if f.exists():
            m = json.loads(f.read_text())["metrics"]
            lines.append(f"{label:48s} final {wc:3s}: AUC {m['add_auc']:.4f}  MPPE {m['mppe_cm']:.2f} cm  pos {100*m['object_position_error_m']:.2f} cm  rot {m['object_orientation_error_deg']:.2f} deg  "
                         f"SP-SR {m['spider_sr_uncentered']:.2f}  MP-SR {m['maniptrans_sr']:.2f}  per-body AUC {[round(x, 3) for x in m['add_auc_per_body']]}")
ax[0].legend(fontsize=7); fig.tight_layout(); (S / "_summaries").mkdir(exist_ok=True); fig.savefig(S / f"_summaries/ep{ep}_compare.png", dpi=105)
(S / f"_summaries/ep{ep}_compare.txt").write_text(f"ep{ep}: official evaluator, final checkpoint (96.8M) and best per-checkpoint 64-world AUC\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
