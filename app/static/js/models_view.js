import { api } from "./api.js";
import { state, indexModels, el, fmtBytes, toast } from "./state.js";

let onCreateFromModel = null;
export function setCreateHandler(fn) { onCreateFromModel = fn; }

export async function loadModels(rescan = false) {
  const warn = document.getElementById("models-warn");
  warn.textContent = "scanning…";
  try {
    const res = rescan ? await api.rescan() : await api.listModels();
    state.models = res.models || [];
    indexModels();
    warn.textContent = (res.warnings || []).length
      ? `${res.warnings.length} warning(s): ${res.warnings.slice(0, 2).join("; ")}`
      : "";
    renderModels();
  } catch (e) {
    warn.textContent = "";
    toast(`scan failed: ${e.message}`, "error");
  }
}

function renderModels() {
  const filter = (document.getElementById("model-filter").value || "").toLowerCase();
  const tbody = document.querySelector("#models-table tbody");
  tbody.innerHTML = "";
  const rows = state.models.filter((m) => {
    if (!filter) return true;
    const arch = m.metadata && m.metadata.architecture ? m.metadata.architecture : "";
    return (m.name + " " + arch).toLowerCase().includes(filter);
  });
  for (const m of rows) {
    const md = m.metadata || {};
    tbody.appendChild(
      el("tr", {},
        el("td", { title: m.display_path }, m.name),
        el("td", {}, md.architecture || "—"),
        el("td", {}, md.n_layers != null ? String(md.n_layers) : "—"),
        el("td", {}, md.context_length != null ? md.context_length.toLocaleString() : "—"),
        el("td", {}, fmtBytes(m.size_bytes)),
        el("td", {}, md.is_moe
          ? el("span", { class: "badge moe" }, `MoE ${md.expert_used_count}/${md.expert_count}`)
          : el("span", { class: "muted" }, "—")),
        el("td", {}, (m.mmproj_candidates || []).length
          ? el("span", { class: "badge" }, "yes") : el("span", { class: "muted" }, "—")),
        el("td", {},
          el("button", { class: "btn small primary", onclick: () => onCreateFromModel && onCreateFromModel(m) }, "New config")),
      )
    );
  }
  if (!rows.length) {
    tbody.appendChild(el("tr", {}, el("td", { colspan: "8", class: "muted" }, "no models found")));
  }
}

export function initModelsView() {
  document.getElementById("btn-rescan").addEventListener("click", () => loadModels(true));
  document.getElementById("model-filter").addEventListener("input", renderModels);
}
