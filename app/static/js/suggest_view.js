import { api } from "./api.js";
import { state, el, toast } from "./state.js";

const CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];

// mount: DOM node; getModelPath: () => string; onApply: (kind, suggestion) => void;
// getMmproj: () => string|null (counted against VRAM when set)
export function renderSuggest(mount, getModelPath, onApply, getMmproj) {
  mount.innerHTML = "";
  const box = el("div", { class: "suggest-box" });

  const ctxInput = el("input", { type: "number", placeholder: "auto (max)", style: "width:120px" });
  const ctk = selectFrom(CACHE_TYPES, "f16");
  const ctv = selectFrom(CACHE_TYPES, "f16");
  const mode = selectFrom(["free", "total", "manual"], "free");
  const manual = el("input", { type: "number", placeholder: "VRAM MiB", style: "width:110px", disabled: "disabled" });
  mode.addEventListener("change", () => (manual.disabled = mode.value !== "manual"));

  const controls = el("div", { class: "suggest-controls" },
    fld("context", ctxInput),
    fld("ctk", ctk), fld("ctv", ctv),
    fld("VRAM budget", mode), fld("manual MiB", manual),
    el("button", { class: "btn primary", onclick: run }, "Suggest"),
  );
  box.appendChild(controls);

  const result = el("div", {});
  box.appendChild(result);
  mount.appendChild(box);

  async function run() {
    const model_path = getModelPath();
    if (!model_path) { toast("select a model first", "error"); return; }
    result.innerHTML = "computing…";
    try {
      const sugg = await api.suggest({
        model_path,
        mmproj_path: getMmproj ? (getMmproj() || null) : null,
        context: ctxInput.value ? Number(ctxInput.value) : null,
        ctk: ctk.value, ctv: ctv.value,
        vram_budget_mode: mode.value,
        manual_vram_mib: manual.value ? Number(manual.value) : null,
      });
      renderResult(result, sugg, onApply);
    } catch (e) {
      result.innerHTML = "";
      result.appendChild(el("div", { class: "warn" }, `suggest failed: ${e.message}`));
    }
  }
}

function renderResult(mount, sugg, onApply) {
  mount.innerHTML = "";
  if (!sugg.explicit || sugg.explicit.c === undefined) {
    mount.appendChild(el("div", { class: "warn" }, (sugg.warnings || []).join("; ") || "no suggestion"));
    return;
  }
  const b = sugg.breakdown || {};
  const lines = [
    `VRAM budget:   ${b.budget_mib} MiB (of ${b.vram_mib}, headroom ${b.headroom_mib} + reserve ${b.compute_reserve_mib})`,
    `layers:        offload ${b.ngl} / ${b.offloadable_layers}  (${b.fits_all_layers ? "ALL fit" : "partial"})`,
    `per-layer:     ${b.bytes_per_layer_mib} MiB    KV total: ${b.kv_total_mib_at_ctx} MiB @ ctx ${sugg.explicit.c}`,
    b.mmproj_mib ? `mmproj:        ${b.mmproj_mib} MiB (GPU-offloaded projector)` : null,
    `est. VRAM use: ${b.estimated_vram_used_mib} MiB`,
    b.is_moe ? `MoE:           ${b.expert_used_count}/${b.expert_count} experts active (full weights resident)` : null,
    b.moe_offload && b.moe_offload["n-cpu-moe"] != null
      ? `MoE offload:   n-cpu-moe=${b.moe_offload["n-cpu-moe"]} (experts of first N layers -> CPU RAM)`
      : null,
  ].filter(Boolean);
  mount.appendChild(el("div", { class: "breakdown" }, lines.join("\n")));
  if (b.moe_hint) mount.appendChild(el("div", { class: "warn" }, b.moe_hint));
  for (const w of sugg.warnings || []) {
    if (w !== b.moe_hint) mount.appendChild(el("div", { class: "warn" }, w));
  }
  mount.appendChild(el("div", { class: "apply-row" },
    el("button", { class: "btn primary", onclick: () => onApply("explicit", sugg) },
      `Apply explicit (` +
      (sugg.explicit["n-cpu-moe"] != null
        ? `n-cpu-moe ${sugg.explicit["n-cpu-moe"]}`
        : `-ngl ${sugg.explicit.ngl}`) +
      `, -c ${sugg.explicit.c})`),
    el("button", { class: "btn", onclick: () => onApply("fit", sugg) },
      `Apply fit (fitc ${sugg.fit.fitc}, fitt ${sugg.fit.fitt})`),
  ));
}

function selectFrom(opts, val) {
  const s = el("select", {});
  for (const o of opts) s.appendChild(el("option", { value: o }, o));
  s.value = val;
  return s;
}
function fld(label, input) {
  return el("div", { class: "fld" }, el("span", {}, label), input);
}
