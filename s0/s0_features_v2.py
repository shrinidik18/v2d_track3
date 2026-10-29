"""S0 extras for the Track 3 evaluation episodes, from the Ego2Sim v2 package (2026-10-27); s0_features_eval.py on v2 inputs.
Pickup frames come from the v2 object trajectories (no extra cleaning pass).

Rays: his raw bundle gives the camera trajectory in the bundle world; the bundle's tracked object maps exactly (scale 1, 0 cm)
into his simulator frame, which gives the camera in the simulator frame. Ray = camera centre
through each of his simulator-frame MANO keypoints (wrist + fingertips); valid = the bundle's hand-valid flag.
Image features: the frozen spatial-softmax encoders (trained on the public episodes) on the cam_a frames; hand crops around
his keypoints projected into the image. Normalised with the same statistics as the public features.
Shape: 32-d BPS of his reconstructed meshes (same basis / projection as s0_features.py); surface points; pickup frames from
the cleaned object trajectories.
    run_in_env.sh --gpu N /env/s0/s0_features_v2.py <episodes...>
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/env/s0")
import s0_features as F  # noqa: E402

PK = Path("/data/outputs/ego2sim_recon_v2_pkg")
VID = Path("/data/outputs/rian_eval_compare")
OUT = Path("/data/outputs/rl_train_req/s0_features_v2")


def fit_sim(src, dst):
    ms, md = src.mean(0), dst.mean(0); A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); Rm = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (A ** 2).sum()
    return s, Rm, md - s * Rm @ ms


def bps_from(paths):
    import trimesh
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(512, 3)); basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    basis *= 0.2 * rng.random((512, 1)) ** (1 / 3)
    proj = np.random.default_rng(11).normal(size=(512, 32)) / np.sqrt(512)
    codes, surf = [], []
    for p in paths:
        m = trimesh.load(p, process=False, force="mesh")
        pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
        codes.append((cKDTree(pts - (m.bounds[0] + m.bounds[1]) / 2).query(basis)[0] @ proj) / 0.05)
        surf.append(trimesh.sample.sample_surface(m, 96, seed=0)[0].astype(np.float32))
    return np.asarray(codes, np.float32), np.stack(surf)


def pickup_frames(P, Q):
    from scipy.spatial.transform import Rotation as R
    T = len(P); onsets = []
    for b in range(P.shape[1]):
        rot = R.from_quat(Q[:, b][:, [1, 2, 3, 0]]); mv = np.zeros(T, bool)
        for f in range(T):
            g = min(f + 10, T - 1)
            mv[f] = np.linalg.norm(P[f:g + 1, b] - P[f, b], axis=1).max() > 0.01 or np.degrees((rot[f:g + 1] * rot[f].inv()).magnitude()).max() > 5
        onsets += [f for f in range(T) if mv[f] and not mv[max(0, f - 10):f].any()]
    return sorted(set(onsets)), sorted({c for o in onsets for c in range(max(0, o - 10), max(1, o - 4))})


def main(eps):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    glob_m, crop_m = F.build_models(); st = torch.load(F.OUT / "encoders.pt")
    glob_m.load_state_dict(st["glob"]); crop_m.load_state_dict(st["crop"]); glob_m.to(dev).eval(); crop_m.to(dev).eval()
    with torch.no_grad():                                                   # normalisation statistics of the public features
        allf = []
        for ep in F.PUBLIC:
            c = np.load(F.CACHE / f"ep{ep}.npz")
            fg = glob_m["enc"](torch.from_numpy(c["imgs"]).to(dev).permute(0, 3, 1, 2).float() / 255).cpu().numpy()
            fc = crop_m["enc"](torch.from_numpy(c["crops"]).to(dev).flatten(0, 1).permute(0, 3, 1, 2).float() / 255).cpu().numpy()
            fc = fc.reshape(len(fg), 2, 32) * c["kp_ok"][..., None]
            allf.append(np.concatenate([fg, fc.reshape(len(fg), 64), c["kp_ok"].astype(np.float32)], 1))
        allf = np.concatenate(allf); mu, sd = allf.mean(0), allf.std(0) + 1e-3; mu[-2:], sd[-2:] = 0.0, 1.0
    import cv2
    for ep in eps:
        t = np.load(PK / f"trajectories/ep{ep}.npz", allow_pickle=True)
        names = [str(n) for n in t["object_names"]]
        bdir = sorted(glob.glob(str(PK / f"raw/ep{ep}_*")))[0]; z = np.load(bdir + "/result.npz", allow_pickle=True)
        T = t["mano_left_joints"].shape[0]
        b = names.index(bdir.split(f"ep{ep}_", 1)[1])                       # the bundle's tracked object maps exactly into the sim frame
        src = z["object_to_world_transform"][:T, :3, 3]; dst = t["object_position"][:, b]
        s, Rm, tt = fit_sim(src, dst)
        res = np.linalg.norm(s * src @ Rm.T + tt - dst, axis=1)
        c2w = z["camera_to_world_transform"][:T].astype(float)
        cam = s * c2w[:, :3, 3] @ Rm.T + tt                                  # camera centre, simulator frame
        J = np.stack([t[f"mano_{sd_}_joints"] for sd_ in ("left", "right")], 1)  # (T,2,21,3) simulator frame
        d = J[:, :, F.KP] - cam[:, None, None]; d /= np.linalg.norm(d, axis=-1, keepdims=True)
        valid = np.stack([z[f"hand_{sd_}_is_valid"][:T] for sd_ in ("left", "right")], 1).astype(np.float32)
        # image features: frames + hand crops around the projected keypoints
        Jraw = ((J - tt) @ Rm) / s                                            # back to the bundle world
        Rcw = np.transpose(c2w[:, :3, :3], (0, 2, 1))
        Jc = np.einsum("tij,thkj->thki", Rcw, Jraw - c2w[:, None, None, :3, 3])
        fx, fy, cx, cy = [float(v) for v in z["camera_intrinsics"]]
        uv = np.stack([fx * Jc[..., 0] / Jc[..., 2] + cx, fy * Jc[..., 1] / Jc[..., 2] + cy], -1)
        cap = cv2.VideoCapture(str(VID / f"ep{ep}/ego_video_cam_a.mp4"))
        imgs = np.zeros((T, F.GH, F.GW, 3), np.uint8); crops = np.zeros((T, 2, F.CROP, F.CROP, 3), np.uint8); kp_ok = np.zeros((T, 2), bool)
        for f in range(T):
            okf, fr = cap.read()
            if not okf: break
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); W, H = im.size
            imgs[f] = np.asarray(im.resize((F.GW, F.GH), Image.BILINEAR))
            for h in range(2):
                if valid[f, h] < 0.5 or (Jc[f, h, :, 2] <= 0.05).any(): continue
                u = uv[f, h]; ctr = u.mean(0); half = max(24.0, 0.7 * float(np.ptp(u, 0).max()))
                if not (-0.1 * W < ctr[0] < 1.1 * W and -0.1 * H < ctr[1] < 1.1 * H): continue
                cr = im.crop(tuple(int(round(v)) for v in (ctr[0] - half, ctr[1] - half, ctr[0] + half, ctr[1] + half))).resize((F.CROP, F.CROP), Image.BILINEAR)
                if h == 0: cr = cr.transpose(Image.FLIP_LEFT_RIGHT)
                crops[f, h] = np.asarray(cr); kp_ok[f, h] = True
        with torch.no_grad():
            fg = glob_m["enc"](torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255).cpu().numpy()
            fc = crop_m["enc"](torch.from_numpy(crops).to(dev).flatten(0, 1).permute(0, 3, 1, 2).float() / 255).cpu().numpy()
        fc = fc.reshape(T, 2, 32) * kp_ok[..., None]
        feats = (np.concatenate([fg, fc.reshape(T, 64), kp_ok.astype(np.float32)], 1) - mu) / sd
        bps, surf = bps_from([PK / "assets" / f"{n}.obj" for n in names])
        onsets, cand = pickup_frames(t["object_position"], t["object_wxyz"])
        np.savez(OUT / f"episode_{ep:06d}.npz", ray_origin=cam.astype(np.float32), ray_dir=d.astype(np.float32),
                 ray_valid=np.repeat(valid[:, :, None], 6, 2), image_features=feats.astype(np.float32), bps=bps,
                 object_names=np.asarray(names), object_surface_points=surf, pickup_onsets=np.asarray(onsets, np.int32),
                 pickup_reset_frames=np.asarray(cand, np.int32),
                 note="eval episode, Ego2Sim v2: rays from the v2 camera track + v2 simulator-frame MANO keypoints")
        print(f"ep{ep}: T={T} objects {names} | bundle->sim scale {s:.3f}, object residual median {100*np.median(res):.2f} cm | "
              f"hands valid {valid.mean(0).round(2).tolist()} crops {kp_ok.mean(0).round(2).tolist()} | pickup onsets {onsets}", flush=True)


if __name__ == "__main__":
    main([int(e) for e in sys.argv[1:]])
