"""Monitor for the S0 runs: 64-world evaluation + zoomed policy video at every checkpoint, graphs and progress.txt."""
import glob, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/usr1/e2s/retarget_env")
import monitor_training as mt

mt.T = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0")
runs = dict(a.split(":") for a in sys.argv[1:]); done = set(); last = False
while True:
    for run, gpu in runs.items():
        for f in sorted(glob.glob(str(mt.T / run / "policy_*.safetensors"))):
            step = int(re.findall(r"policy_(\d+)\.", f)[0])
            if (run, step) in done or (mt.T / run / f"eval/policy_{step}.mp4").exists():
                done.add((run, step)); continue
            time.sleep(20)
            subprocess.run(["/usr1/e2s/retarget_env/s0/s0_eval.sh", str(mt.T / run), str(step), gpu, "64"], env={**os.environ, "RENDER": "1"})
            done.add((run, step))
        mt.graphs(run)
    mt.write_txt(list(runs))
    if last: break
    last = all("completed" in mt.progress_line(r) for r in runs)
    time.sleep(60 if not last else 1)
