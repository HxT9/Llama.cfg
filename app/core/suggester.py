"""VRAM/RAM-aware suggestion engine.

Given GGUF metadata and a VRAM budget, propose how many layers to offload
(`-ngl`) and what context (`-c`) to use, and an equivalent llama.cpp `fit`
configuration. The math is deliberately approximate (per-layer weight cost is
derived from file size); the `fit` variant lets llama.cpp do exact placement.

MoE note: a GGUF's file size already includes ALL experts, and every expert
weight must be resident on its device. `expert_used_count` only reduces compute
per token, not the memory footprint. So the offload math uses the full file
size; when layers don't all fit we recommend `--cpu-moe`/`--n-cpu-moe` to push
expert weights to CPU RAM instead.
"""
from __future__ import annotations

from app.config import (
    CONTEXT_STEPS,
    DEFAULT_COMPUTE_RESERVE_MIB,
    DEFAULT_HEADROOM_FRAC,
    KV_BYTES_PER_ELEM,
)
from app.models import GgufMetadata, Suggestion

MIB = 1024 * 1024


def _bpe(quant: str) -> float:
    return KV_BYTES_PER_ELEM.get((quant or "f16").lower(), 2.0)


def suggest(
    meta: GgufMetadata,
    vram_mib: int,
    *,
    context: int | None = None,
    ctk: str = "f16",
    ctv: str = "f16",
    headroom_frac: float = DEFAULT_HEADROOM_FRAC,
    compute_reserve_mib: int = DEFAULT_COMPUTE_RESERVE_MIB,
    mmproj_bytes: int = 0,
    expert_fraction: float | None = None,
) -> Suggestion:
    warnings: list[str] = []

    n_layers = meta.n_layers
    if not n_layers or not meta.file_size_bytes:
        warnings.append(
            "GGUF metadata missing layer count or file size; cannot compute offload."
        )
        return Suggestion(warnings=warnings)

    offloadable = n_layers + 1  # repeating blocks + output/embedding layer
    bytes_per_layer = meta.file_size_bytes / offloadable

    # the multimodal projector (if loaded with --mmproj) is GPU-offloaded by
    # default, so its weights eat into the layer-offload budget.
    mmproj_bytes = max(0, int(mmproj_bytes))
    mmproj_mib = mmproj_bytes / MIB

    headroom_mib = vram_mib * headroom_frac
    budget_mib = vram_mib - headroom_mib - compute_reserve_mib
    budget_bytes = max(0.0, budget_mib * MIB - mmproj_bytes)

    n_head_kv = meta.n_head_kv
    head_dim = meta.head_dim
    kv_per_layer_token = 0.0
    if n_head_kv and head_dim:
        kv_per_layer_token = n_head_kv * head_dim * (_bpe(ctk) + _bpe(ctv))
    else:
        warnings.append("KV cache size unknown (missing head metadata); ignored in math.")

    def kv_bytes(n: int, c: int) -> float:
        return n * c * kv_per_layer_token

    def max_ngl(c: int) -> int:
        for n in range(offloadable, -1, -1):
            if n * bytes_per_layer + kv_bytes(n, c) <= budget_bytes:
                return n
        return 0

    train_ctx = meta.context_length or None

    # --- choose context + ngl ----------------------------------------------
    if context:
        chosen_ctx = min(context, train_ctx) if train_ctx else context
        if train_ctx and context > train_ctx:
            warnings.append(
                f"requested context {context} exceeds model train context "
                f"{train_ctx}; capped (use RoPE scaling to go beyond)."
            )
        ngl = max_ngl(chosen_ctx)
    else:
        candidates = sorted(
            {c for c in CONTEXT_STEPS if (not train_ctx or c <= train_ctx)},
            reverse=True,
        )
        if train_ctx:
            candidates = sorted(set(candidates) | {train_ctx}, reverse=True)
        if not candidates:
            candidates = [8192]
        chosen_ctx = candidates[-1]
        ngl = max_ngl(chosen_ctx)
        for c in candidates:  # largest ctx that still offloads everything
            if max_ngl(c) >= offloadable:
                chosen_ctx, ngl = c, offloadable
                break

    ngl = min(ngl, offloadable)
    fits_all = ngl >= offloadable

    used_mib = (ngl * bytes_per_layer + kv_bytes(ngl, chosen_ctx) + mmproj_bytes) / MIB

    # --- MoE expert offload (better than dropping whole layers) -------------
    # When a MoE model doesn't fully fit, keep ALL layers on GPU (attention +
    # KV stay fast) and push the bulk expert weights of the first N layers to
    # CPU RAM via --n-cpu-moe (N == n_layers means every layer's experts).
    moe_offload: dict = {}
    moe_hint = None
    if meta.is_moe and not fits_all and expert_fraction and 0 < expert_fraction < 1:
        expert_bytes = meta.file_size_bytes * expert_fraction
        non_expert_bytes = meta.file_size_bytes - expert_bytes
        per_layer_expert = expert_bytes / n_layers
        kv_all = kv_bytes(offloadable, chosen_ctx)
        avail_for_experts = budget_bytes - non_expert_bytes - kv_all
        if per_layer_expert > 0 and avail_for_experts > 0:
            layers_experts_on_gpu = min(n_layers, int(avail_for_experts // per_layer_expert))
            n_cpu_moe = n_layers - layers_experts_on_gpu
            ngl = offloadable                       # all layers on GPU
            fits_all = True                         # weights placed (experts spilled)
            if n_cpu_moe > 0:
                moe_offload = {"n-cpu-moe": n_cpu_moe}
            used_mib = (
                non_expert_bytes
                + layers_experts_on_gpu * per_layer_expert
                + kv_all + mmproj_bytes
            ) / MIB
        else:
            # not even attention layers + KV fit with every layer's experts on CPU
            moe_hint = (
                "Even with --n-cpu-moe set to all layers, the non-expert weights "
                "plus KV cache exceed VRAM. Lower context or quantize the KV cache."
            )
            warnings.append(moe_hint)
    elif meta.is_moe and not fits_all:
        moe_hint = (
            "MoE model and not all layers fit: use --n-cpu-moe N to keep the expert "
            "weights of the first N layers in CPU RAM (N up to all layers), freeing "
            "VRAM for attention layers + context."
        )
        warnings.append(moe_hint)

    explicit = {"ngl": ngl, "c": chosen_ctx, "ctk": ctk, "ctv": ctv, **moe_offload}
    # Let --fit fully own device placement (it handles MoE expert offload
    # itself). Pinning n-cpu-moe over-constrains it and leaves VRAM underused,
    # so the fit config carries only fit/fitc/fitt + the KV quant choice.
    fit = {
        "fit": "on",
        "fitc": chosen_ctx,
        "fitt": int(headroom_mib + compute_reserve_mib),
        "ctk": ctk,
        "ctv": ctv,
    }
    breakdown = {
        "vram_mib": int(vram_mib),
        "headroom_mib": int(headroom_mib),
        "compute_reserve_mib": int(compute_reserve_mib),
        "budget_mib": int(budget_mib),
        "n_layers": n_layers,
        "offloadable_layers": offloadable,
        "bytes_per_layer_mib": round(bytes_per_layer / MIB, 2),
        "kv_per_layer_per_token_bytes": round(kv_per_layer_token, 2),
        "kv_total_mib_at_ctx": round(kv_bytes(ngl, chosen_ctx) / MIB, 2),
        "mmproj_mib": round(mmproj_mib, 2),
        "estimated_vram_used_mib": round(used_mib, 2),
        "fits_all_layers": fits_all,
        "is_moe": meta.is_moe,
        "expert_count": meta.expert_count,
        "expert_used_count": meta.expert_used_count,
        "expert_fraction": round(expert_fraction, 3) if expert_fraction else None,
        "moe_offload": moe_offload or None,
        "moe_hint": moe_hint,
    }
    return Suggestion(explicit=explicit, fit=fit, breakdown=breakdown, warnings=warnings)
