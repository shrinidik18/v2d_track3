"""Clean Rian's reconstructed eval-set object trajectories (a copy; his package is untouched).

  stage1  per episode, per object:
          1. rotationally symmetric objects (cups): remove the unobservable spin about the symmetry axis - each frame keeps
             the twist about that axis closest to the previous cleaned frame (rotation about the mesh centroid)
          2. other objects: unwrap look-alike flips - each frame chooses, among {identity, 180 deg about each principal axis}
             applied about the centroid, the orientation closest to the previous cleaned frame (only when that saves >30 deg);
             remaining >25 deg/frame spikes are dropped and re-interpolated (slerp), then a light 5-frame rotation smoothing
          + per frame, an object whose lowest point is below the table top is raised onto it
  stage2  3. frame-0 spawn fix: the poses objects come to rest at in the simulator (released at frame 0, gravity and contacts,
             no object controller, no hands - what the evaluator's physics does anyway) replace frame 0 for objects resting
             >= 5 frames; the correction is kept until the object first moves
             and blended out over 15 frames
  write   cleaned parquet (same layout) under <out>/package, cleaned npz + report per episode

    python clean_refs.py stage1 <ep...> | stage2 <ep...>
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

SRC = Path("/usr1/e2s/v2d_challenge_data/outputs/submission_eval_run/package")       # Rian's package, paths -> /pkg
OUT = Path("/usr1/e2s/v2d_challenge_data/outputs/rian_eval_compare/_cleaned")
PKG = OUT / "package"
TABLE_Z = 0.75
SYMMETRIC = {"blue_cup", "beige_cup"}
_MESH = {}


def mesh_info(name):
    if name not in _MESH:
        m = trimesh.load(SRC / "assets" / f"{name}.obj", process=False, force="mesh")
        pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
        c = pts.mean(0); w, v = np.linalg.eigh(np.cov((pts - c).T))
        gaps = [abs(w[1] - w[2]), abs(w[0] - w[2]), abs(w[0] - w[1])]
        _MESH[name] = dict(c=c, axes=v, sym_axis=v[:, int(np.argmin(gaps))], hull=np.asarray(m.convex_hull.vertices))
    return _MESH[name]


def parquet_path(root, ep):
    return next((root / f"sim_reference/processed/sequence_id=ep{ep}/robot_name=sharpa_wave").glob("*.parquet"))


def rotate_about_centroid(P, rot_old, rot_new, c):
    """Positions so the mesh centroid stays where it was when the orientation changes rot_old -> rot_new."""
    return P + rot_old.apply(c) - rot_new.apply(c)


def jumps(rot):
    return np.degrees((rot[1:] * rot[:-1].inv()).magnitude())


def unwrap(rot, S):
    """Continuity: each frame takes, among rot[t]*s (s in S), the orientation closest to the previous cleaned frame
    (only when that saves >30 deg over rot[t] itself)."""
    out = [rot[0]]
    for t in range(1, len(rot)):
        cand = [rot[t] * s for s in S]
        dist = [np.degrees((cc * out[-1].inv()).magnitude()) for cc in cand]
        k = int(np.argmin(dist))
        out.append(cand[k] if dist[0] - dist[k] > 30.0 else cand[0])
    return R.concatenate(out)


def majority_branch(new, rot, S, axis=None):
    """Frame 0 may itself be the flipped hypothesis: apply the constant flip s that makes the cleaned track agree with the raw
    track on most frames (orientation within 45 deg; for symmetric objects only the symmetry-axis direction counts)."""
    best, score = new, -1.0
    for s in S:
        cand = new * s
        if axis is None:
            agree = np.mean(np.degrees((cand * rot.inv()).magnitude()) < 45.0)
        else:
            agree = np.mean(np.einsum("ij,ij->i", cand.apply(axis), rot.apply(axis)) > np.cos(np.radians(45.0)))
        if agree > score:
            best, score = cand, agree
    return best, float(score)


def despike_and_smooth(new, P):
    T = len(new)
    good = np.ones(T, bool); last = 0                                     # rotation spikes: >25 deg/frame from the last good frame
    for t in range(1, T):
        bad_rot = np.degrees((new[t] * new[last].inv()).magnitude()) > 25.0 * (t - last)
        bad_pos = np.linalg.norm(P[t] - P[last]) > 0.05 * (t - last)          # position spikes: >5 cm/frame (1 m/s)
        if bad_rot or bad_pos:
            good[t] = False
        else:
            last = t
    if good.sum() >= 2 and not good.all():
        idx = np.flatnonzero(good); tt = np.clip(np.arange(T), idx[0], idx[-1])
        fill = Slerp(idx, new[idx])(tt)
        new = R.concatenate([new[t] if good[t] else fill[t] for t in range(T)])
        P = np.stack([np.interp(tt, idx, P[idx, k]) for k in range(3)], 1)
    q = new.as_quat()                                                     # hemisphere-consistent, then 5-tap smoothing
    for t in range(1, T):
        if q[t] @ q[t - 1] < 0: q[t] *= -1
    w = np.array([1, 4, 6, 4, 1], float); sm = np.zeros_like(q)
    for t in range(T):
        idx = np.arange(max(0, t - 2), min(T, t + 3)); sm[t] = (w[idx - t + 2][:, None] * q[idx]).sum(0)
    return R.from_quat(sm / np.linalg.norm(sm, axis=1, keepdims=True)), P, int((~good).sum())


def clean_object(P, Q, name):
    info = mesh_info(name); c = info["c"]; T = len(P)
    rot = R.from_quat(Q[:, [1, 2, 3, 0]])
    if name in SYMMETRIC:
        a = info["sym_axis"]; out = [rot[0]]
        for t in range(1, T):                                             # 1. remove spin about the symmetry axis
            d = (rot[t].inv() * out[-1]).as_quat()
            tw = np.r_[(d[:3] @ a) * a, d[3]]; tw /= np.linalg.norm(tw)
            out.append(rot[t] * R.from_quat(tw))
        new = R.concatenate(out)
        perp = [k for k in range(3) if abs(info["axes"][:, k] @ a) < 0.5]
        S = [R.identity()] + [R.from_rotvec(np.pi * info["axes"][:, k]) for k in perp]
        new = unwrap(new, S); new, agree = majority_branch(new, rot, S, axis=a)
    else:                                                                 # 2. unwrap look-alike flips
        S = [R.identity()] + [R.from_rotvec(np.pi * info["axes"][:, k]) for k in range(3)]
        new = unwrap(rot, S); new, agree = majority_branch(new, rot, S)
    Pn = rotate_about_centroid(P, rot, new, c)
    new, Pn, n_spikes = despike_and_smooth(new, Pn)
    low = np.array([(new[t].apply(info["hull"]) + Pn[t])[:, 2].min() for t in range(T)])
    lift = np.maximum(0.0, TABLE_Z - low)
    Pn[:, 2] += lift
    return Pn, new, dict(raw_jumps_gt20=int((jumps(rot) > 20).sum()), clean_jumps_gt20=int((jumps(new) > 20).sum()),
                         raw_flips_gt150=int((jumps(rot) > 150).sum()), clean_flips_gt150=int((jumps(new) > 150).sum()),
                         raw_p95_deg=float(np.percentile(jumps(rot), 95)), clean_p95_deg=float(np.percentile(jumps(new), 95)),
                         agreement_with_raw=agree, spike_frames_interpolated=n_spikes,
                         frames_lifted_onto_table=int((lift > 1e-3).sum()), max_lift_cm=float(100 * lift.max()))


def set_col(t, name, value):
    i = t.column_names.index(name)
    return t.set_column(i, t.schema.field(i), pa.array([value], type=t.schema.field(i).type))


def write(ep, P, rots, report):
    p = parquet_path(PKG, ep); t = pq.read_table(p)
    q = np.stack([r.as_quat()[:, [3, 0, 1, 2]] for r in rots], 1)          # (T, B, 4) wxyz
    t = set_col(t, "object_body_position", P.tolist()); t = set_col(t, "object_body_wxyz", q.tolist())
    t = set_col(t, "object_root_position", P[:, 0].tolist()); t = set_col(t, "object_root_axis_angle", rots[0].as_rotvec().tolist())
    pq.write_table(t, p)
    np.savez(OUT / f"ep{ep}_cleaned.npz", object_names=np.asarray(report["objects"]), object_position=P, object_wxyz=q)
    (OUT / f"ep{ep}_report.json").write_text(json.dumps(report, indent=1) + "\n")


def load(root, ep):
    r = pq.read_table(parquet_path(root, ep)).to_pylist()[0]
    names = [str(n) for n in r["object_body_names"]]
    T = len(r["robot_left_wrist_position"]); B = len(names)
    return names, np.asarray(r["object_body_position"], float).reshape(T, B, 3), np.asarray(r["object_body_wxyz"], float).reshape(T, B, 4)


def stage1(eps):
    if not PKG.exists():
        shutil.copytree(SRC, PKG)
    for ep in eps:
        names, P, Q = load(SRC, ep)
        Ps, rots, objs = [], [], {}
        for b, n in enumerate(names):
            Pn, rn, rep = clean_object(P[:, b], Q[:, b], n); Ps.append(Pn); rots.append(rn); objs[n] = rep
        write(ep, np.stack(Ps, 1), rots, {"episode": ep, "objects": names, "stage1": objs})
        print(f"ep{ep}: " + "; ".join(f"{n} jumps>20 {o['raw_jumps_gt20']}->{o['clean_jumps_gt20']}, flips {o['raw_flips_gt150']}->{o['clean_flips_gt150']}, "
                                      f"agree {o['agreement_with_raw']:.2f}, spikes {o['spike_frames_interpolated']}, lifted {o['frames_lifted_onto_table']} fr (max {o['max_lift_cm']:.1f} cm)" for n, o in objs.items()), flush=True)


def stage2(eps):
    for ep in eps:
        rep = json.loads((OUT / f"ep{ep}_report.json").read_text())
        settled = json.loads((OUT / f"ep{ep}_settle.json").read_text())
        names, P, Q = load(PKG, ep); T = len(P)
        Pn, rots, fix = P.copy(), [], {}
        for b, n in enumerate(names):
            rot = R.from_quat(Q[:, b][:, [1, 2, 3, 0]])
            sp = np.asarray(settled["position"][b]); sr = R.from_quat(np.asarray(settled["quat_xyzw"][b]))
            st = R.from_quat(np.asarray(settled["start_quat_xyzw"][b]))     # simulator body frame at the reference frame-0 pose
            dp = sp - np.asarray(settled["start_position"][b]); dr = sr * st.inv()
            moved = np.flatnonzero((np.linalg.norm(P[:, b] - P[0, b], axis=1) > 0.01) | (np.degrees((rot * rot[0].inv()).magnitude()) > 5))
            onset = int(moved[0]) if len(moved) else T
            wgt = np.clip(1.0 - (np.arange(T) - onset) / 15.0, 0.0, 1.0)
            small = (np.linalg.norm(dp) < 0.003 and np.degrees(dr.magnitude()) < 2) or onset < 5   # only objects resting >= 5 frames
            if small or np.degrees(dr.magnitude()) > 45.0:                  # toppled in the settle: keep the reference pose
                small = True
                wgt[:] = 0.0
            rot_ok = np.degrees(dr.magnitude()) <= 45.0                      # larger settle rotations: correct position only
            corr = R.from_rotvec((dr.as_rotvec() if rot_ok else np.zeros(3))[None] * wgt[:, None])
            new = corr * rot
            Pn[:, b] = P[:, b] + dp[None] * wgt[:, None]
            rots.append(new)
            fix[n] = {"frame0_shift_cm": float(100 * np.linalg.norm(dp)), "frame0_rotation_deg": float(np.degrees(dr.magnitude())),
                      "held_until_frame": onset, "applied": not small, "rotation_applied": bool(rot_ok and not small)}
        rep["stage2_spawn_fix"] = fix
        write(ep, Pn, rots, rep)
        print(f"ep{ep}: " + "; ".join(f"{n} {('shift+rot' if f['rotation_applied'] else 'shift') if f['applied'] else 'unchanged'} {f['frame0_shift_cm']:.1f} cm / {f['frame0_rotation_deg']:.1f} deg (held to frame {f['held_until_frame']})"
                                      for n, f in fix.items()), flush=True)


if __name__ == "__main__":
    eps = [int(e) for e in sys.argv[2:]]
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]](eps)
