"""Monitor one experiment folder (training_s0/<exp>): 64-world eval + video for every checkpoint of every phase, graphs, and
<exp>/progress.txt. Exits once every phase has completed and its last checkpoint is processed.
    python s0_monitor_exp.py <exp> <gpu> <n_phases>"""
import glob, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/usr1/e2s/retarget_env")
import monitor_training as mt

exp, gpu, n_phases = sys.argv[1], sys.argv[2], int(sys.argv[3])
mt.T = Path("/usr1/e2s/v2d_challenge_data/outputs/rl_train_req/training_s0") / exp
done, last = set(), False
while True:
    phases = sorted(p.name for p in mt.T.glob("p*_*M") if p.is_dir())
    for run in phases:
        for f in sorted(glob.glob(str(mt.T / run / "policy_*.safetensors")), key=lambda x: int(re.findall(r"policy_(\d+)\.", x)[0])):
            step = int(re.findall(r"policy_(\d+)\.", f)[0])
            if (run, step) in done or (mt.T / run / f"eval/policy_{step}.mp4").exists():
                done.add((run, step)); continue
            time.sleep(20)
            subprocess.run(["/usr1/e2s/retarget_env/s0/s0_eval.sh", str(mt.T / run), str(step), gpu, "64"], env={**os.environ, "RENDER": "1"})
            done.add((run, step))
        mt.graphs(run)
    mt.write_txt(phases)
    if last: break
    last = len(phases) >= n_phases and all("completed" in mt.progress_line(r) for r in phases)
    time.sleep(60 if not last else 1)
