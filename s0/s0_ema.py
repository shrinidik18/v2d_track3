"""Exponential moving average of the FlashSAC actor weights, saved as the policy checkpoint (S0_EMA=<tau>).

Every learner step (one environment collection) the averaged actor moves toward the live actor by tau; policy_*.safetensors
then holds the averaged weights (batch-norm statistics from the live actor), while state_*.safetensors keeps the raw learner
so training and resume are unchanged. tau = 0.001 averages over ~1000 collections (~1-2M environment steps).
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

import flash_chord.training.flash_sac.checkpoint as _ckpt
import flash_chord.training.flash_sac.runner as _runner

_STATE: dict = {"ema": None, "tau": 0.001}


@jax.jit
def _ema_update(ema, params, tau):
    return jax.tree_util.tree_map(lambda e, p: e + tau * (p - e), ema, params)


def install(tau: float) -> None:
    _STATE["tau"] = float(tau)
    orig_compile = _runner._compile_executables          # wraps the ahead-of-time compiled learner step

    def compile_executables(*args, **kwargs):
        executables = orig_compile(*args, **kwargs)
        orig_consume = executables.consume_transition

        def consume_transition(learner, *a, **k):
            out = orig_consume(learner, *a, **k)
            params = out[0].actor.params
            if _STATE["ema"] is None:
                _STATE["ema"] = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), params)
            else:
                _STATE["ema"] = _ema_update(_STATE["ema"], params, jnp.float32(_STATE["tau"]))
            return out
        return dataclasses.replace(executables, consume_transition=consume_transition)

    _runner._compile_executables = compile_executables
    orig_state = _ckpt.actor_inference_state

    def actor_inference_state(learner):
        live = orig_state(learner)
        if _STATE["ema"] is None:
            return live
        return dataclasses.replace(live, params=_STATE["ema"])

    _ckpt.actor_inference_state = actor_inference_state
    print(f"[s0] actor EMA installed (tau={tau}); policy checkpoints hold the averaged weights", flush=True)
