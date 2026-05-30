// Client-side VRAM usage estimate. Mirrors app/core/suggester.py so the
// editor can show a live figure without a round-trip. Approximate by design.
const MIB = 1048576;
const KV_BYTES_PER_ELEM = {
  f32: 4, f16: 2, bf16: 2, q8_0: 1.06, q5_1: 0.75, q5_0: 0.69,
  q4_1: 0.63, q4_0: 0.56, iq4_nl: 0.56,
};
const bpe = (q) => KV_BYTES_PER_ELEM[(q || "f16").toLowerCase()] ?? 2;
const num = (v) => { const n = Number(v); return Number.isFinite(n) ? n : 0; };

function parseNgl(v, offloadable) {
  if (v == null || v === "") return offloadable;       // unset -> assume all (auto)
  const s = String(v).toLowerCase();
  if (s === "all" || s === "auto" || s === "-1" || s === "999") return offloadable;
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? Math.max(0, Math.min(offloadable, n)) : offloadable;
}

// meta: GgufMetadata; flags: {key:value}; totalVramMib: number|null
export function computeEstimate(meta, flags, totalVramMib) {
  if (!meta || !meta.n_layers || !meta.file_size_bytes) return null;
  const offloadable = meta.n_layers + 1;
  const bytesPerLayer = meta.file_size_bytes / offloadable;
  const fitOn = !!flags.fit && /^(on|true|1)$/i.test(String(flags.fit));

  if (fitOn) {
    const fitt = num(flags.fitt);
    return {
      mode: "fit",
      vramMib: totalVramMib ? Math.max(0, totalVramMib - fitt) : null,
      totalVramMib,
      contextUsed: num(flags.fitc) || null,
      fitt,
    };
  }

  const ngl = parseNgl(flags.ngl, offloadable);
  const ctx = num(flags.c) || meta.context_length || 4096;
  const nCpuMoe = num(flags["n-cpu-moe"]);
  const perLayerExpert =
    meta.is_moe && meta.expert_fraction
      ? (meta.expert_fraction * meta.file_size_bytes) / meta.n_layers
      : 0;

  // per-layer slice of all weights, minus experts pushed to CPU for the
  // first N (offloaded) layers
  let gpuWeight = ngl * bytesPerLayer - Math.min(nCpuMoe, ngl) * perLayerExpert;
  gpuWeight = Math.max(0, gpuWeight);

  // KV cache: use the per-layer profile (sliding-window + hybrid aware) when
  // available, else a simple global-layer fallback.
  const bk = bpe(flags.ctk), bv = bpe(flags.ctv);
  let kvTotal;
  if (meta.kv_k_global != null) {
    const gk = meta.kv_k_global, gv = meta.kv_v_global || 0;
    const sk = meta.kv_k_swa || 0, sv = meta.kv_v_swa || 0, win = meta.kv_window;
    kvTotal = ctx * (gk * bk + gv * bv) + Math.min(win || ctx, ctx) * (sk * bk + sv * bv);
  } else if (meta.n_head_kv && meta.head_dim) {
    const fa = meta.full_attention_interval;
    const nG = fa && fa > 1 ? Math.max(1, Math.round(meta.n_layers / fa)) : meta.n_layers;
    kvTotal = ctx * nG * meta.n_head_kv * meta.head_dim * (bk + bv);
  } else {
    kvTotal = 0;
  }
  const kvFrac = meta.n_layers ? Math.min(ngl, meta.n_layers) / meta.n_layers : 0;
  const kv = kvTotal * kvFrac;

  return {
    mode: "explicit",
    vramMib: (gpuWeight + kv) / MIB,
    gpuWeightMib: gpuWeight / MIB,
    kvMib: kv / MIB,
    cpuWeightMib: Math.max(0, meta.file_size_bytes - gpuWeight) / MIB,
    totalVramMib,
    contextUsed: ctx,
    nglUsed: ngl,
    offloadable,
  };
}
