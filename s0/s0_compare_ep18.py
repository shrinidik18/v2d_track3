"""ep18 (our hands): baseline vs S0 (from scratch) vs S0.5 (initialised from ep40's 25M S0 weights)."""
import glob, json, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req")
RUNS = {"baseline (hand reward to 80M)": ("training/ep18_ours_p1_80M", "training/ep18_ours_p2_ft_objonly"),
        "S0 (hand reward off at 55M)": ("training_s0/ep18_s0_p1_80M", "training_s0/ep18_s0_p2_ft_objonly"),
        "S0.5 (init: ep40 S0 @25M)": ("training_s0/ep18_s0.5_from_ep40_25M_p1_80M", "training_s0/ep18_s0.5_from_ep40_25M_p2_ft_objonly"),
        "S0-fix (relative action, no hand terms, rays)": ("training_s0/ep18_s0fix_p1_80M", "training_s0/ep18_s0fix_p2_ft_objonly"),
        "S0.5-fix (same, init: ep40 @25M)": ("training_s0/ep18_s0.5fix_from_ep40_25M_p1_80M", "training_s0/ep18_s0.5fix_from_ep40_25M_p2_ft_objonly"),
        "S0-contact (+ contact reward, gated VOC)": ("training_s0/ep18_s0contact_p1_80M", "training_s0/ep18_s0contact_p2_ft_objonly"),
        "S0.5-contact (same, init: ep40 @25M)": ("training_s0/ep18_s0.5contact_from_ep40_25M_p1_80M", "training_s0/ep18_s0.5contact_from_ep40_25M_p2_ft_objonly")}
KEYS = [("add_auc", "ADD AUC"), ("object_position_error_m", "object position error (m)"),
        ("object_orientation_error_deg", "object orientation error (deg)"), ("mppe_cm", "MPPE (cm)")]
fig, ax = plt.subplots(1, 4, figsize=(24, 6)); lines = []
for label, dirs in RUNS.items():
    pts = []
    for d in dirs:
        for f in glob.glob(str(R / d / "eval/metrics_*.json")):
            pts.append((int(re.findall(r"metrics_(\d+)", f)[0]), json.loads(Path(f).read_text())["metrics"]))
    pts.sort()
    for col, (k, title) in enumerate(KEYS):
        if pts: ax[col].plot([s / 1e6 for s, _ in pts], [m[k] for _, m in pts], marker="o", ms=3, label=label)
        ax[col].set_title(f"ep18: {title}"); ax[col].set_xlabel("env steps (M)"); ax[col].grid(alpha=.3)
    for wc in ("w1", "w64"):
        f = R / dirs[1] / f"eval_final/metrics_96776192_{wc}.json"
        if f.exists():
            m = json.loads(f.read_text())["metrics"]
            lines.append(f"{label:32s} {wc:3s}: AUC {m['add_auc']:.4f}  MPPE {m['mppe_cm']:.2f} cm  pos {100*m['object_position_error_m']:.2f} cm  "
                         f"rot {m['object_orientation_error_deg']:.2f} deg  SP-SR {m['spider_sr_uncentered']:.2f}  MP-SR {m['maniptrans_sr']:.2f}  "
                         f"per-body AUC {[round(x, 3) for x in m['add_auc_per_body']]}")
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(R / "training_s0/ep18_compare.png", dpi=110)
(R / "training_s0/ep18_compare.txt").write_text("ep18, final checkpoint (96.8M), official evaluator; per-body = [mini_sweeper, mini_dust_pan]\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
