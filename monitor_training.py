"""Watch FlashSAC runs: score + render each new checkpoint, and keep progress.txt and graphs up to date.

    python monitor_training.py <run_name>:<gpu> [<run_name>:<gpu> ...]      (runs live under rl_train_req/training/)
"""
import glob, json, re, subprocess, sys, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training")
ENV = Path("/usr1/e2s/retarget_env")
TRAIN = ["reward/total_mean_per_env_step", "episode/return_mean_per_completed_episode",
         "episode/reference_progress_mean_per_completed_episode", "episode/length_mean_per_completed_episode",
         "curriculum/voc_scale", "replay/fill_fraction"]
EVAL = ["evaluation/chord_sr", "evaluation/add_auc", "evaluation/object_position_error_m",
        "evaluation/object_orientation_error_deg", "evaluation/mean_per_point_error_cm", "evaluation/mppe_cm"]


def rows(run):
    p = T / run / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def evals(run):
    out = []
    for f in sorted(glob.glob(str(T / run / "eval/metrics_*.json")), key=lambda x: int(re.findall(r"_(\d+)\.json", x)[0])):
        d = json.loads(Path(f).read_text()); step = int(re.findall(r"_(\d+)\.json", f)[0])
        flat = d.get("scalars", d) if isinstance(d, dict) else {}
        out.append((step, flat))
    return out


def find(d, key):
    if key in d: return d[key]
    short = key.split("/")[-1]
    for k, v in d.items():
        if isinstance(v, dict):
            r = find(v, key)
            if r is not None: return r
        elif k == short and not isinstance(v, (dict, list)): return v
    return None


def progress_line(run):
    t = (T / f"{run}.log"); last = ""
    if t.exists():
        m = [l for l in t.read_text(errors="ignore").splitlines() if l.startswith(("training progress", "completed"))]
        last = m[-1] if m else ""
    return last


def graphs(run):
    r = rows(run)
    if r:
        fig, ax = plt.subplots(2, 3, figsize=(15, 7)); ax = ax.ravel()
        tr_err = [k for k in r[-1] if k.startswith("tracking_error/")][:0]
        for a, k in zip(ax, TRAIN):
            xs = [x["step"] / 1e6 for x in r if k in x]; ys = [x[k] for x in r if k in x]
            a.plot(xs, ys, lw=1); a.set_title(k, fontsize=9); a.set_xlabel("env steps (M)"); a.grid(alpha=.3)
        fig.suptitle(f"{run}: training"); fig.tight_layout(); fig.savefig(T / run / "training_curves.png", dpi=110); plt.close(fig)
        errs = sorted({k for x in r for k in x if k.startswith("tracking_error/")})
        if errs:
            fig, ax = plt.subplots(figsize=(9, 5))
            for k in errs: ax.plot([x["step"] / 1e6 for x in r if k in x], [x[k] for x in r if k in x], lw=1, label=k.split("/")[1][:34])
            ax.set_yscale("log"); ax.legend(fontsize=6); ax.set_xlabel("env steps (M)"); ax.grid(alpha=.3); ax.set_title(f"{run}: tracking errors (training)")
            fig.savefig(T / run / "tracking_errors_training.png", dpi=110); plt.close(fig)
    e = evals(run)
    if e:
        fig, ax = plt.subplots(2, 3, figsize=(15, 7)); ax = ax.ravel()
        for a, k in zip(ax, EVAL):
            pts = [(s / 1e6, find(d, k)) for s, d in e if find(d, k) is not None]
            if pts: a.plot(*zip(*pts), marker="o")
            cap = {"evaluation/mppe_cm": 10.0, "evaluation/mean_per_point_error_cm": 10.0, "evaluation/object_position_error_m": 0.10}.get(k)
            if pts and cap is not None:                   # fixed 0-cap axis; off-scale points (e.g. an exploded world) drawn at the top
                over = [(x, cap) for x, v in pts if v > cap]
                if over: a.scatter(*zip(*over), marker="^", color="red", zorder=3, label="off scale")
                a.set_ylim(0, cap)
            a.set_title(k, fontsize=9); a.set_xlabel("env steps (M)"); a.grid(alpha=.3)
        fig.suptitle(f"{run}: evaluation of each checkpoint (64 worlds, unassisted full sequence)")
        fig.tight_layout(); fig.savefig(T / run / "eval_curves.png", dpi=110); plt.close(fig)


def write_txt(runs):
    lines = [f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for run in runs:
        lines += [f"== {run}", progress_line(run)]
        r = rows(run)
        if r:
            x = r[-1]; lines.append("  latest training log (step %d): " % x["step"] + ", ".join(
                f"{k.split('/')[-1][:28]}={x[k]:.4g}" for k in TRAIN if k in x))
            terms = {k: v for k, v in x.items() if k.startswith("termination/")}
            if terms: lines.append("  terminations: " + ", ".join(f"{k.split('/')[1].replace('_fraction_per_completed_episode','')}={v:.2f}" for k, v in terms.items()))
        for s, d in evals(run):
            vals = {k.split("/")[-1]: find(d, k) for k in EVAL}
            lines.append(f"  eval @ {s/1e6:6.1f}M: " + ", ".join(f"{k}={v:.4g}" for k, v in vals.items() if v is not None))
        lines.append("")
    (T / "progress.txt").write_text("\n".join(lines))


def main():
    runs = dict(a.split(":") for a in sys.argv[1:]); done = set(); last_pass = False
    while True:
        for run, gpu in runs.items():
            for f in sorted(glob.glob(str(T / run / "policy_*.safetensors"))):
                step = int(re.findall(r"policy_(\d+)\.", f)[0])
                if (run, step) in done or (T / run / f"eval/policy_{step}.mp4").exists(): done.add((run, step)); continue
                time.sleep(20)                                   # let the checkpoint finish writing
                subprocess.run([str(ENV / "run_eval.sh"), str(T / run), str(step), gpu, "64"], env={"RENDER_FRAMES": "750", **__import__("os").environ})
                done.add((run, step))
            graphs(run)
        write_txt(list(runs))
        if last_pass: break
        last_pass = all("completed" in progress_line(r) for r in runs)      # one more scan after training ends, for the final checkpoint
        time.sleep(60 if not last_pass else 1)


if __name__ == "__main__":
    main()
