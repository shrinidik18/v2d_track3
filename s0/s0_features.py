"""S0 extra policy inputs from the egocentric video: hand rays, a light spatial image encoder, and an object shape code.

Stages (run in the retargeting env, `run_in_env.sh --gpu N /env/s0/s0_features.py <stage>`):
  labels  per public episode: camera centre and hand keypoints in the GT world (from our registered reconstruction),
          2D keypoint labels, object-mask box labels, downsampled frames and hand crops -> cache
  check   overlay the projected keypoints on frames + distance of the teammate's wrist to our wrist ray (sanity)
  train   train the spatial-softmax encoders (global frame + hand crop) on the cached labels
  export  per-episode extras npz: rays, image features, BPS shape code

Rays: origin = camera centre, direction = through the hand keypoint. Our hands were registered with one similarity
(scale s, yaw R_w, offset) that also moves the camera, so the ray direction is exactly the 2D image evidence and does
not depend on the (unreliable) depth.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

D = Path("/data/outputs")
EPS_DIR = D / "track3_handfix"
SIDE = D / "rl_train_req/episodes"
OUT = D / "rl_train_req/s0_features"
CACHE = OUT / "cache"
PUBLIC = [0, 2, 11, 12, 13, 18, 22, 23, 24, 26, 27, 31, 40, 41, 42, 43, 44, 45]
KP = [0, 4, 8, 12, 16, 20]                    # MANO wrist + fingertips (OpenPose order)
GW, GH = 256, 192                             # global encoder input
CROP = 64


_MANO = None


def _mano():
    global _MANO
    if _MANO is None:
        import torch  # noqa: F401
        from manotorch.manolayer import ManoLayer
        _MANO = ManoLayer(rot_mode="axisang", use_pca=False, side="right", center_idx=None,
                          mano_assets_root="/weights/hamer/_DATA/data")
    return _MANO


def hand_joints_cam(ep, T):
    """Image-consistent hand joints in the camera frame: HaMeR fit at cam_t_pre_dz (before the depth shift, which moves
    the hand along the optical axis and breaks its projection), hand_scale 1. Returns (T, 2[left,right], 21, 3), valid."""
    import torch
    epd = EPS_DIR / f"episode_{ep:06d}"
    tracks = json.loads((epd / "hand_tracks.json").read_text())["tracks"]
    J = np.zeros((T, 2, 21, 3)); ok = np.zeros((T, 2), bool); intr = {}
    for tr in tracks:
        h = 1 if tr["is_right"] else 0
        for f in sorted((epd / "hamer_refined_simple" / str(tr["object_id"])).glob("*.json")):
            rec = json.loads(f.read_text()); t = int(rec["frame_idx"])
            if t >= T or rec.get("interpolated", False):
                continue
            pose = np.concatenate([rec["mano"]["global_orient"], rec["mano"]["hand_pose"]]).astype(np.float32)
            out = _mano()(torch.from_numpy(pose)[None], torch.from_numpy(np.asarray(rec["mano"]["betas"], np.float32))[None])
            j = out.joints[0].detach().numpy().astype(float)
            if not rec["is_right"]:
                j[:, 0] *= -1
            J[t, h] = j + np.asarray(rec["diagnostics"]["cam_t_pre_dz"], float)
            ok[t, h] = J[t, h, :, 2].min() > 0.05
            intr = rec["camera"]["intrinsics"]
    return J, ok, intr


def episode_geometry(ep):
    side = np.load(SIDE / f"episode_{ep:06d}/hand_trajectories.npz", allow_pickle=True)
    meta = json.loads((SIDE / f"episode_{ep:06d}/meta.json").read_text())
    reg = meta["registration"]
    R_w, off, s = np.asarray(reg["R_w"], float), np.asarray(reg["offset"], float), float(reg["scale_s"])
    z = np.load(EPS_DIR / f"episode_{ep:06d}/result_slam_gravity_aligned/result.npz", allow_pickle=True)
    c2w = z["camera_to_world_transform"].astype(float)
    T = side["time_s"].shape[0]
    cam_gt = (c2w[:T, :3, 3] / s) @ R_w.T + off                   # camera centre in the GT world
    Jc, ok, intr = hand_joints_cam(ep, T)
    K = intr if intr else json.loads((EPS_DIR / f"episode_{ep:06d}/intrinsics_stable.json").read_text())
    dirs_cam = Jc / np.linalg.norm(Jc, axis=-1, keepdims=True)
    R_gc = np.einsum("ij,tjk->tik", R_w, c2w[:T, :3, :3])          # camera -> GT rotation per frame
    dirs_gt = np.einsum("tij,thkj->thki", R_gc, dirs_cam)
    uv = np.stack([K["fx"] * Jc[..., 0] / Jc[..., 2] + K["cx"], K["fy"] * Jc[..., 1] / Jc[..., 2] + K["cy"]], -1)
    return dict(T=T, cam=cam_gt, dirs=dirs_gt, uv=uv, ok=ok, K=K, side=side, meta=meta,
                joints_gt={sd: side[f"mano_{sd}_joints"].astype(float) for sd in ("left", "right")})


def stage_labels():
    CACHE.mkdir(parents=True, exist_ok=True)
    for ep in PUBLIC:
        g = episode_geometry(ep)
        T, K = g["T"], g["K"]
        epd = EPS_DIR / f"episode_{ep:06d}"
        obj_id = json.loads((epd / "object_track.json").read_text())["object_id"]
        imgs = np.zeros((T, GH, GW, 3), np.uint8)
        kp2d = np.zeros((T, 2, 6, 2), np.float32); kp_ok = np.zeros((T, 2), bool)
        crops = np.zeros((T, 2, CROP, CROP, 3), np.uint8); crop_kp = np.zeros((T, 2, 21, 2), np.float32)
        objbox = np.zeros((T, 4), np.float32); obj_ok = np.zeros(T, bool)
        for t in range(T):
            im = Image.open(epd / "frames" / f"{t:06d}.png").convert("RGB")
            W, H = im.size
            imgs[t] = np.asarray(im.resize((GW, GH), Image.BILINEAR))
            for h, sd in enumerate(("left", "right")):
                if not g["ok"][t, h]:
                    continue
                uv = g["uv"][t, h]
                kp2d[t, h] = uv[KP] / [W, H]
                kp_ok[t, h] = (uv[KP, 0] > -0.1 * W).all() and (uv[KP, 0] < 1.1 * W).all() and (uv[KP, 1] > -0.1 * H).all() and (uv[KP, 1] < 1.1 * H).all()
                c = uv.mean(0); half = max(24.0, 0.7 * float(np.ptp(uv, 0).max()))
                box = (c[0] - half, c[1] - half, c[0] + half, c[1] + half)
                cr = im.crop(tuple(int(round(v)) for v in box)).resize((CROP, CROP), Image.BILINEAR)
                ck = (uv - [box[0], box[1]]) / (2 * half)
                if sd == "left":                                  # share one encoder: mirror left hands
                    cr = cr.transpose(Image.FLIP_LEFT_RIGHT); ck[:, 0] = 1 - ck[:, 0]
                crops[t, h] = np.asarray(cr); crop_kp[t, h] = ck
            mp = epd / "masks" / str(obj_id) / f"{t:06d}.png"
            if mp.exists():
                m = np.asarray(Image.open(mp)) > 0
                if m.ndim == 3: m = m[..., 0]
                ys, xs = np.nonzero(m)
                if len(xs) > 50:
                    objbox[t] = [xs.mean() / W, ys.mean() / H, np.ptp(xs) / W, np.ptp(ys) / H]; obj_ok[t] = True
        np.savez(CACHE / f"ep{ep}.npz", imgs=imgs, kp2d=kp2d, kp_ok=kp_ok, crops=crops, crop_kp=crop_kp, objbox=objbox, obj_ok=obj_ok)
        print(f"ep{ep}: {T} frames, hands ok L/R {kp_ok[:, 0].mean():.2f}/{kp_ok[:, 1].mean():.2f}, object box {obj_ok.mean():.2f}", flush=True)


def stage_check():
    from scipy.optimize import minimize_scalar  # noqa: F401  (scipy available in the env)
    for ep in (40, 41):
        g = episode_geometry(ep)
        im = Image.open(EPS_DIR / f"episode_{ep:06d}/frames/000060.png").convert("RGB")
        from PIL import ImageDraw
        dr = ImageDraw.Draw(im)
        for h, col in ((0, (255, 60, 60)), (1, (60, 160, 255))):
            for u, v in g["uv"][60, h]:
                dr.ellipse([u - 7, v - 7, u + 7, v + 7], outline=col, width=3)
        im.save(OUT / f"check_projection_ep{ep}_f60.jpg", quality=85)
        # teammate wrist vs our wrist ray (perpendicular distance), in the GT frame
        tm = SIDE / f"episode_{ep:06d}/teammate_hands_meta.json"
        import pyarrow.parquet as pq
        p = sorted((D / f"rl_train_req/assets/human_motion_data/ego_recon/processed/sequence_id=episode_{ep:06d}_teammate/robot_name=sharpa_wave").glob("*.parquet"))[0]
        row = pq.read_table(p).to_pylist()[0]
        for sd in ("left", "right"):
            mj = np.asarray(row[f"mano_{sd}_joints"], float)                     # teammate MANO joints, GT frame
            h = 0 if sd == "left" else 1
            o = g["cam"]; dvec = g["dirs"][:, h, 0]
            w = mj[:, 0] - o
            perp = np.linalg.norm(w - (w * dvec).sum(1, keepdims=True) * dvec, axis=1)
            ok = g["ok"][:, h]
            print(f"ep{ep} {sd}: teammate wrist distance to our wrist ray median {100*np.median(perp[ok]):.1f} cm, p90 {100*np.percentile(perp[ok], 90):.1f} cm "
                  f"(our registered 3D wrist vs teammate wrist median {100*np.median(np.linalg.norm(mj[:, 0]-g['joints_gt'][sd][:, 0], axis=1)[ok]):.1f} cm) {tm.exists()}")


def build_models():
    import torch
    import torch.nn as nn

    class SpatialSoftmax(nn.Module):
        def forward(self, x):
            b, c, h, w = x.shape
            p = torch.softmax(x.flatten(2), -1).view(b, c, h, w)
            ys = torch.linspace(-1, 1, h, device=x.device); xs = torch.linspace(-1, 1, w, device=x.device)
            return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], 1)      # (b, 2c): x then y

    def enc(cin, k, mid=32):
        return nn.Sequential(nn.Conv2d(cin, mid, 5, 2, 2), nn.ReLU(), nn.Conv2d(mid, 64, 3, 2, 1), nn.ReLU(),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, k, 3, 1, 1), SpatialSoftmax())
    glob = nn.ModuleDict({"enc": enc(3, 32), "head": nn.Linear(64, 24 + 4)})     # 2 hands x 6 kp x (u,v) + object box
    crop = nn.ModuleDict({"enc": enc(3, 16, 32), "head": nn.Linear(32, 42)})     # 21 kp x (u,v) in the crop
    return glob, crop


def load_cache():
    data = [np.load(CACHE / f"ep{ep}.npz") for ep in PUBLIC]
    cat = lambda k: np.concatenate([d[k] for d in data])  # noqa: E731
    return {k: cat(k) for k in ("imgs", "kp2d", "kp_ok", "crops", "crop_kp", "objbox", "obj_ok")}


def stage_train():
    import torch
    dev = "cuda"
    c = load_cache()
    n = len(c["imgs"]); rng = np.random.default_rng(0); val = rng.random(n) < 0.1
    glob, crop = build_models(); glob.to(dev); crop.to(dev)
    opt = torch.optim.Adam(list(glob.parameters()) + list(crop.parameters()), 1e-3)
    X = torch.from_numpy(c["imgs"]).to(dev); C = torch.from_numpy(c["crops"]).to(dev)
    kp = torch.from_numpy(c["kp2d"]).to(dev).view(n, 24) * 2 - 1; kpok = torch.from_numpy(c["kp_ok"]).to(dev)
    ob = torch.from_numpy(c["objbox"]).to(dev); obok = torch.from_numpy(c["obj_ok"]).to(dev)
    ck = torch.from_numpy(c["crop_kp"]).to(dev) * 2 - 1
    tr_idx = np.flatnonzero(~val); va_idx = np.flatnonzero(val)

    def step(idx, train):
        i = torch.as_tensor(idx, device=dev)
        x = X[i].permute(0, 3, 1, 2).float() / 255
        if train: x = x * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, 1, device=dev))
        out = glob["head"](glob["enc"](x))
        m_kp = kpok[i].repeat_interleave(12, 1).float()
        l_kp = (((out[:, :24] - kp[i]) ** 2) * m_kp).sum() / m_kp.sum().clamp(min=1)
        l_ob = (((out[:, 24:] - (ob[i] * 2 - 1)) ** 2).mean(1) * obok[i]).sum() / obok[i].sum().clamp(min=1)
        hc = C[i].flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        okc = kpok[i].flatten()
        oc = crop["head"](crop["enc"](hc)).view(-1, 21, 2)
        l_c = (((oc - ck[i].flatten(0, 1)) ** 2).mean((1, 2)) * okc).sum() / okc.sum().clamp(min=1)
        return l_kp + l_ob + l_c, (l_kp.item(), l_ob.item(), l_c.item())
    for epoch in range(40):
        glob.train(); crop.train(); rng.shuffle(tr_idx)
        for b in range(0, len(tr_idx), 64):
            loss, _ = step(tr_idx[b:b + 64], True); opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 4 or epoch == 39:
            glob.eval(); crop.eval()
            with torch.no_grad():
                parts = np.mean([step(va_idx[b:b + 128], False)[1] for b in range(0, len(va_idx), 128)], 0)
            # rmse in normalised image units -> pixels of a 2028-wide frame (keypoints), crop fraction (crop keypoints)
            print(f"epoch {epoch+1}: val hand-kp rmse {np.sqrt(parts[0]/2)*1014:.0f} px, object-box mse {parts[1]:.4f}, crop-kp rmse {np.sqrt(parts[2])*32:.1f} px (of 64)", flush=True)
    torch.save({"glob": glob.state_dict(), "crop": crop.state_dict()}, OUT / "encoders.pt")
    n_params = sum(p.numel() for m in (glob, crop) for p in m["enc"].parameters())
    print(f"saved encoders ({n_params/1e3:.0f}k encoder parameters)")


def bps_code(names):
    import trimesh
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(512, 3)); basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    basis *= 0.2 * rng.random((512, 1)) ** (1 / 3)                    # uniform in a 20 cm ball
    proj = np.random.default_rng(11).normal(size=(512, 32)) / np.sqrt(512)   # fixed random projection to 32-d
    from scipy.spatial import cKDTree
    codes = []
    for n in names:
        m = trimesh.load(D / f"rl_train_req/assets/human_motion_data/ego_recon/processed/{n}.obj", process=False, force="mesh")
        pts = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
        pts = pts - (m.bounds[0] + m.bounds[1]) / 2
        dist = cKDTree(pts).query(basis)[0]
        codes.append((dist @ proj) / 0.05)
    return np.asarray(codes, np.float32)


def stage_export(eps):
    import torch
    dev = "cuda"
    glob, crop = build_models(); st = torch.load(OUT / "encoders.pt")
    glob.load_state_dict(st["glob"]); crop.load_state_dict(st["crop"]); glob.to(dev).eval(); crop.to(dev).eval()
    feats_all = {}
    with torch.no_grad():
        for ep in PUBLIC:
            c = np.load(CACHE / f"ep{ep}.npz")
            x = torch.from_numpy(c["imgs"]).to(dev).permute(0, 3, 1, 2).float() / 255
            fg = glob["enc"](x).cpu().numpy()
            hc = torch.from_numpy(c["crops"]).to(dev).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
            fc = crop["enc"](hc).cpu().numpy().reshape(len(fg), 2, 32) * c["kp_ok"][..., None]
            feats_all[ep] = np.concatenate([fg, fc.reshape(len(fg), 64), c["kp_ok"].astype(np.float32)], 1)
    allf = np.concatenate(list(feats_all.values())); mu, sd = allf.mean(0), allf.std(0) + 1e-3
    mu[-2:], sd[-2:] = 0.0, 1.0                                         # keep the two validity flags as 0/1
    for ep in eps:
        g = episode_geometry(ep)
        rays_d = np.zeros((g["T"], 2, 6, 3), np.float32); valid = np.zeros((g["T"], 2, 6), np.float32)
        for h in range(2):
            rays_d[:, h] = g["dirs"][:, h, KP]
            valid[:, h] = g["ok"][:, h, None]
        names = [str(n) for n in g["side"]["object_names"]]
        np.savez(OUT / f"episode_{ep:06d}.npz", ray_origin=g["cam"].astype(np.float32), ray_dir=rays_d, ray_valid=valid,
                 image_features=((feats_all[ep] - mu) / sd).astype(np.float32), bps=bps_code(names), object_names=np.asarray(names),
                 note="ray sides ordered (left, right); keypoints wrist, thumb..pinky tips; image features = 64 global "
                      "spatial-softmax + 2x32 hand-crop (left, right) + 2 hand-valid flags, normalised")
        print(f"exported ep{ep}: T={g['T']}, image features {feats_all[ep].shape[1]}, objects {names}")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "labels": stage_labels()
    elif stage == "check": stage_check()
    elif stage == "train": stage_train()
    elif stage == "export": stage_export([int(e) for e in sys.argv[2].split(",")])
