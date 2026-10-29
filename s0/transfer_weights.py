"""Transfer a trained FlashSAC learner state between episodes whose observation layouts differ (e.g. 1 -> 2 objects).

Every tensor of the target template keeps its shape. Tensors that match in shape are copied from the source. Tensors with
an observation (actor) or critic-input axis are remapped block by block by name: per-object blocks map the source object
onto the same-named object in the target; rows of a new object get zero weights (ignored at first) and borrow the matched
object's normalisation statistics. Per-world buffers and integer counters stay as in the template (fresh run at step ~0).

  python transfer_weights.py SRC_STATE TEMPLATE_STATE OUT_STATE --src-objects a --dst-objects b,a
"""
import argparse
import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

K_FUTURE = 10
PER_OBJECT = {"object_position_e_m": 3, "object_orientation_e_wxyz": 4, "command_object_relative_position_m": 3,
              "command_object_relative_orientation_wxyz": 4, "object_shape_bps32": 32, "object_body_velocity_w_mps_radps": 6}


def read(path):
    with safe_open(path, "np") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, f.metadata()


def obs_map(src_blocks, dst_blocks, n_src, n_dst, obj_src_of_dst):
    """Return (index, proxy, new) arrays over destination rows: source row (-1 for new rows), proxy source row for stats."""
    src = {b["name"]: (b["start"], b["end"]) for b in src_blocks}
    total = dst_blocks[-1]["end"] if dst_blocks else 0
    idx = np.full(total, -1); proxy = np.full(total, -1)
    for b in dst_blocks:
        name, ds, de = b["name"], b["start"], b["end"]
        if name not in src:
            raise KeyError(f"destination block {name} not in source")
        ss, se = src[name]
        wd, ws = de - ds, se - ss
        if wd == ws:
            idx[ds:de] = proxy[ds:de] = np.arange(ss, se); continue
        if name.startswith("object_future_relative_pose"):
            unit = 7
            for k in range(K_FUTURE):
                for o in range(n_dst):
                    so = obj_src_of_dst[o]; p = 0 if so < 0 else so
                    d0 = ds + (k * n_dst + o) * unit; s0 = ss + (k * n_src + p) * unit
                    proxy[d0:d0 + unit] = np.arange(s0, s0 + unit)
                    if so >= 0: idx[d0:d0 + unit] = np.arange(s0, s0 + unit)
            continue
        unit = PER_OBJECT.get(name) or (ws // n_src if name.startswith("contact_") else None)
        if unit is None or wd != unit * n_dst or ws != unit * n_src:
            raise ValueError(f"cannot map block {name}: widths {ws} -> {wd}")
        for o in range(n_dst):
            so = obj_src_of_dst[o]; p = 0 if so < 0 else so
            d0, s0 = ds + o * unit, ss + p * unit
            proxy[d0:d0 + unit] = np.arange(s0, s0 + unit)
            if so >= 0: idx[d0:d0 + unit] = np.arange(s0, s0 + unit)
    return idx, proxy


def context_map(src_names, dst_names, obj_src_of_dst):
    idx, proxy = [], []
    for n in dst_names:
        if n.startswith("object_body_"):
            o = int(n.split("_")[2]); so = obj_src_of_dst[o]; p = 0 if so < 0 else so
            sname = n.replace(f"object_body_{o}_", f"object_body_{p}_", 1)
            proxy.append(src_names.index(sname)); idx.append(src_names.index(sname) if so >= 0 else -1)
        else:
            i = src_names.index(n); idx.append(i); proxy.append(i)
    return np.asarray(idx), np.asarray(proxy)


def remap(t, axis, idx, proxy, is_weight):
    out = np.take(t, proxy, axis=axis).copy()
    if is_weight:
        new = np.flatnonzero(idx < 0)
        sl = [slice(None)] * t.ndim; sl[axis] = new; out[tuple(sl)] = 0.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("template"); p.add_argument("out")
    p.add_argument("--src-objects", required=True); p.add_argument("--dst-objects", required=True)
    a = p.parse_args()
    so, dobj = a.src_objects.split(","), a.dst_objects.split(",")
    obj_src_of_dst = [so.index(n) if n in so else -1 for n in dobj]
    S, smd = read(a.src); D, dmd = read(a.template)
    sp, dp = json.loads(smd["user.policy_schema_json"]), json.loads(dmd["user.policy_schema_json"])
    sc, dc = json.loads(smd["user.critic_schema_json"]), json.loads(dmd["user.critic_schema_json"])
    a_idx, a_px = obs_map(sp["observation"]["blocks"], dp["observation"]["blocks"], len(so), len(dobj), obj_src_of_dst)
    c_idx_ctx, c_px_ctx = context_map([b["name"] for b in sc["context"]["blocks"]], [b["name"] for b in dc["context"]["blocks"]], obj_src_of_dst)
    s_obs, s_ctx = sc["observation"]["dimension"], sc["context"]["dimension"]
    n_act = dc["action"]["dimension"]
    c_idx = np.concatenate([a_idx, np.where(c_idx_ctx < 0, -1, c_idx_ctx + s_obs), s_obs + s_ctx + np.arange(n_act)])
    c_px = np.concatenate([a_px, c_px_ctx + s_obs, s_obs + s_ctx + np.arange(n_act)])
    dims = {(sp["observation"]["dimension"], dp["observation"]["dimension"]): (a_idx, a_px),
            (sc["input_dimension"], dc["input_dimension"]): (c_idx, c_px)}
    out, log = {}, {"copied": 0, "remapped": 0, "template": 0}
    for k, t in D.items():
        s = S[k]
        if s.shape == t.shape and s.dtype == t.dtype and (t.dtype.kind == "f"):
            out[k] = s; log["copied"] += 1; continue
        done = False
        if s.ndim == t.ndim and s.dtype == t.dtype and t.dtype.kind == "f":
            diff = [i for i in range(t.ndim) if s.shape[i] != t.shape[i]]
            if len(diff) == 1 and (s.shape[diff[0]], t.shape[diff[0]]) in dims:
                ax = diff[0]; idx, px = dims[(s.shape[ax], t.shape[ax])]
                lead = 1 if (t.ndim >= 2 and t.shape[0] == 2 and ax != 0) else 0      # critic ensemble axis
                is_weight = (t.ndim - lead) >= 2
                out[k] = remap(s, ax, idx, px, is_weight).astype(t.dtype); log["remapped"] += 1; done = True
        if not done:
            out[k] = t; log["template"] += 1
    save_file(out, a.out, metadata=dmd)
    print(json.dumps({**log, "new_rows_actor": int((a_idx < 0).sum()), "new_rows_critic": int((c_idx < 0).sum()),
                      "object_map": dict(zip(dobj, obj_src_of_dst))}))


if __name__ == "__main__":
    main()
