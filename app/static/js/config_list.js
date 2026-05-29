import { api } from "./api.js";
import { state, el, toast } from "./state.js";
import { renderEditor } from "./flag_editor.js";

export async function loadConfigs(keepSelection = true) {
  state.configs = await api.configs();
  if (!keepSelection || !state.configs.find((c) => c.id === state.selectedConfigId)) {
    state.selectedConfigId = state.configs.length ? state.configs[0].id : null;
  }
  renderConfigList();
  renderCurrent();
}

export function renderConfigList() {
  const box = document.getElementById("config-list");
  box.innerHTML = "";
  // group by model_display_path so multiple-per-model is visible
  const groups = {};
  for (const c of state.configs) {
    const key = c.model_display_path || "(no model)";
    (groups[key] = groups[key] || []).push(c);
  }
  for (const [model, items] of Object.entries(groups)) {
    const label = model === "(no model)" ? model : model.split(/[\\/]/).pop();
    box.appendChild(el("div", { class: "cfg-group-label", title: model }, label));
    for (const c of items) {
      box.appendChild(
        el("div", {
          class: "cfg-item" + (c.id === state.selectedConfigId ? " active" : ""),
          onclick: () => { state.selectedConfigId = c.id; renderConfigList(); renderCurrent(); },
        }, c.name || "(unnamed)")
      );
    }
  }
  if (!state.configs.length) {
    box.appendChild(el("div", { class: "empty-hint" }, "No configs yet."));
  }
}

function renderCurrent() {
  const entry = state.configs.find((c) => c.id === state.selectedConfigId);
  renderEditor(entry || null);
}

export async function newBlankConfig() {
  const name = prompt("New config name (INI section):", "my-config");
  if (!name) return;
  const created = await api.createConfig({ name, flags: {} });
  await loadConfigs(false);
  state.selectedConfigId = created.id;
  renderConfigList();
  renderCurrent();
  toast("config created", "ok");
}

export async function createFromModel(model) {
  const base = model.name.replace(/\.gguf$/i, "");
  const name = prompt("New config name (INI section):", base);
  if (!name) return;
  const flags = { "no-mmap": "true" };
  const mmproj = (model.mmproj_candidates || [])[0] || null;
  const created = await api.createConfig({
    name, model_display_path: model.display_path, flags, mmproj,
  });
  await loadConfigs(false);
  state.selectedConfigId = created.id;
  // jump to configs tab
  document.querySelector('.tab[data-tab="configs"]').click();
  renderConfigList();
  renderCurrent();
  toast("config created from model", "ok");
}

export function initConfigList() {
  document.getElementById("btn-new-config").addEventListener("click", newBlankConfig);
}
