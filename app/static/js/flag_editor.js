import { api } from "./api.js";
import { state, el, toast } from "./state.js";
import { renderSuggest } from "./suggest_view.js";

// Working copy of the entry currently being edited.
let current = null;

const CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];

// Curated common flags, using the exact key strings the existing INI uses.
function commonFlags() {
  const ctk = enumValuesFor("cache-type-k") || CACHE_TYPES;
  const spec = enumValuesFor("spec-type") || ["none", "draft-mtp", "draft-simple", "draft-eagle3"];
  return [
    { key: "c", label: "context (-c)", type: "int" },
    { key: "ngl", label: "gpu-layers (-ngl)", type: "int" },
    { key: "ctk", label: "cache-type-k", type: "enum", options: ctk },
    { key: "ctv", label: "cache-type-v", type: "enum", options: ctk },
    { key: "fit", label: "fit", type: "enum", options: ["on", "off"] },
    { key: "fitc", label: "fit-ctx (fitc)", type: "int" },
    { key: "fitt", label: "fit-target (fitt)", type: "string" },
    { key: "no-mmap", label: "no-mmap", type: "bool" },
    { key: "n-cpu-moe", label: "n-cpu-moe", type: "int" },
    { key: "spec-type", label: "spec-type", type: "enum", options: spec },
    { key: "spec-draft-n-max", label: "spec-draft-n-max", type: "int" },
    { key: "chat-template-kwargs", label: "chat-template-kwargs", type: "combo",
      placeholder: '{"enable_thinking": false}' },
  ];
}

// editable-combobox flags remember their values across configs (in settings)
function comboKeys() {
  return commonFlags().filter((s) => s.type === "combo").map((s) => s.key);
}

function presetValues(key) {
  const p = state.settings && state.settings.value_presets;
  return (p && p[key]) || [];
}

function enumValuesFor(canonical) {
  const f = state.flags.find((x) => x.canonical_key === canonical);
  return f && f.enum_values && f.enum_values.length ? f.enum_values : null;
}

export function renderEditor(entry) {
  const box = document.getElementById("config-editor");
  box.innerHTML = "";
  if (!entry) {
    current = null;
    box.appendChild(el("div", { class: "empty-hint" }, "Select a config, or create one from the Models tab."));
    return;
  }
  // deep-clone into working copy
  current = JSON.parse(JSON.stringify(entry));
  current.flags = current.flags || {};

  // header: name + save/delete
  box.appendChild(el("div", { class: "editor-head" },
    el("input", { class: "name", value: current.name, oninput: (e) => (current.name = e.target.value) }),
    el("button", { class: "btn primary", onclick: save }, "Save"),
    el("button", { class: "btn danger", onclick: del }, "Delete"),
  ));

  // model + mmproj selectors
  box.appendChild(el("div", { class: "section-title" }, "Model"));
  box.appendChild(modelSelectors());

  // suggestion panel
  box.appendChild(el("div", { class: "section-title" }, "Suggestion"));
  const suggestMount = el("div", {});
  box.appendChild(suggestMount);
  renderSuggest(suggestMount, () => current.model_display_path, applySuggestion, () => current.mmproj);

  // common flags
  box.appendChild(el("div", { class: "section-title" }, "Common flags"));
  const common = el("div", {});
  for (const spec of commonFlags()) common.appendChild(fieldRow(spec));
  box.appendChild(common);

  // all flags (collapsible)
  box.appendChild(el("div", { class: "section-title" }, "All flags"));
  box.appendChild(allFlagsSection());

  // notes
  box.appendChild(el("div", { class: "section-title" }, "Notes"));
  box.appendChild(el("textarea", {
    rows: "2", style: "width:100%",
    value: current.notes || "",
    oninput: (e) => (current.notes = e.target.value),
  }));
}

function modelSelectors() {
  const wrap = el("div", { class: "field-row" });
  const sel = el("select", {
    onchange: (e) => { current.model_display_path = e.target.value; refreshSuggestMount(); },
  });
  sel.appendChild(el("option", { value: "" }, "— select model —"));
  for (const m of state.models) {
    sel.appendChild(el("option", { value: m.display_path }, `${m.name}`));
  }
  // include current path even if not in scan list
  if (current.model_display_path && !state.modelsByPath[current.model_display_path]) {
    sel.appendChild(el("option", { value: current.model_display_path }, current.model_display_path));
  }
  sel.value = current.model_display_path || "";

  const mmproj = el("select", { onchange: (e) => (current.mmproj = e.target.value || null) });
  mmproj.appendChild(el("option", { value: "" }, "— no mmproj —"));
  const model = state.modelsByPath[current.model_display_path];
  const candidates = (model && model.mmproj_candidates) || [];
  for (const c of candidates) mmproj.appendChild(el("option", { value: c }, c.split(/[\\/]/).pop()));
  if (current.mmproj && !candidates.includes(current.mmproj)) {
    mmproj.appendChild(el("option", { value: current.mmproj }, current.mmproj.split(/[\\/]/).pop()));
  }
  mmproj.value = current.mmproj || "";

  return el("div", {},
    el("div", { class: "field-row" }, el("label", {}, "model"), sel),
    el("div", { class: "field-row" }, el("label", {}, "mmproj"), mmproj),
  );
}

function refreshSuggestMount() {
  // re-render selectors area mmproj options when model changes
  renderEditor(current);
}

// build a single labelled field bound to current.flags[key]
function fieldRow(spec) {
  const enabled = spec.key in current.flags;
  const row = el("div", { class: "field-row" });
  const chk = el("input", {
    type: "checkbox", ...(enabled ? { checked: "checked" } : {}),
    onchange: (e) => {
      if (e.target.checked) current.flags[spec.key] = defaultFor(spec);
      else delete current.flags[spec.key];
      renderEditor(current);
    },
  });
  const label = el("label", { class: "chk-flag" }, chk, spec.label);

  let col;
  const val = current.flags[spec.key] ?? "";
  if (spec.type === "bool") {
    col = el("span", { class: "hint" }, enabled ? "= true" : "(disabled)");
  } else if (spec.type === "enum" || spec.type === "tristate") {
    const sel = el("select", {
      disabled: !enabled, onchange: (e) => (current.flags[spec.key] = e.target.value),
    });
    for (const o of spec.options || []) sel.appendChild(el("option", { value: o }, o));
    sel.value = val;
    col = sel;
  } else if (spec.type === "combo") {
    const listId = `dl-${spec.key}`;
    const input = el("input", {
      type: "text", disabled: !enabled, value: val, list: listId,
      placeholder: spec.placeholder || "", style: "width:100%",
      oninput: (e) => (current.flags[spec.key] = e.target.value),
    });
    const dl = el("datalist", { id: listId });
    for (const v of presetValues(spec.key)) dl.appendChild(el("option", { value: v }));
    col = el("div", {}, input, dl);
  } else {
    col = el("input", {
      type: spec.type === "int" || spec.type === "number" ? "number" : "text",
      disabled: !enabled, value: val,
      oninput: (e) => (current.flags[spec.key] = e.target.value),
    });
  }
  return el("div", { class: "field-row" }, label, col);
}

function defaultFor(spec) {
  if (spec.type === "bool") return "true";
  if (spec.type === "enum" || spec.type === "tristate") return (spec.options || [""])[0];
  return "";
}

function allFlagsSection() {
  const wrap = el("div", {});
  const search = el("input", { class: "filter", placeholder: "search all flags…" });
  wrap.appendChild(el("div", { class: "toolbar" }, search));
  const groupsMount = el("div", {});
  wrap.appendChild(groupsMount);

  const render = () => {
    const q = (search.value || "").toLowerCase();
    groupsMount.innerHTML = "";
    for (const [group, flags] of Object.entries(state.flagsByGroup)) {
      const matched = flags.filter((f) =>
        !q || (f.canonical_key + " " + f.aliases.join(" ") + " " + f.description).toLowerCase().includes(q)
      );
      if (!matched.length) continue;
      const body = el("div", { class: "group-body" });
      for (const f of matched) body.appendChild(allFlagRow(f));
      const det = el("details", { class: "flag-group", ...(q ? { open: "open" } : {}) },
        el("summary", {}, `${group} (${matched.length})`), body);
      groupsMount.appendChild(det);
    }
  };
  search.addEventListener("input", render);
  render();
  return wrap;
}

function allFlagRow(f) {
  const key = f.canonical_key;
  const enabled = key in current.flags;
  const chk = el("input", {
    type: "checkbox", ...(enabled ? { checked: "checked" } : {}),
    onchange: (e) => {
      if (e.target.checked) current.flags[key] = f.takes_value ? "" : "true";
      else delete current.flags[key];
      renderEditor(current);
    },
  });
  const aliasHint = el("span", { class: "hint", title: f.description }, f.aliases.join(", "));
  let input;
  const val = current.flags[key] ?? "";
  if (!f.takes_value) {
    input = el("span", { class: "hint" }, enabled ? "= true" : "");
  } else if ((f.enum_values || []).length) {
    input = el("select", { disabled: !enabled, onchange: (e) => (current.flags[key] = e.target.value) });
    input.appendChild(el("option", { value: "" }, "—"));
    for (const o of f.enum_values) input.appendChild(el("option", { value: o }, o));
    input.value = val;
  } else {
    input = el("input", {
      type: f.value_type === "int" || f.value_type === "number" ? "number" : "text",
      disabled: !enabled, value: val, placeholder: f.default || "",
      oninput: (e) => (current.flags[key] = e.target.value),
    });
  }
  return el("div", { class: "field-row" },
    el("label", { class: "chk-flag" }, chk, key), el("div", {}, input, " ", aliasHint));
}

function applySuggestion(kind, sugg) {
  const f = current.flags;
  if (kind === "explicit") {
    f["ngl"] = String(sugg.explicit.ngl);
    f["c"] = String(sugg.explicit.c);
    f["ctk"] = sugg.explicit.ctk;
    f["ctv"] = sugg.explicit.ctv;
    delete f["fit"]; delete f["fitc"]; delete f["fitt"];
    // MoE expert offload (always n-cpu-moe); clear any stale cpu-moe
    if ("n-cpu-moe" in sugg.explicit) f["n-cpu-moe"] = String(sugg.explicit["n-cpu-moe"]);
    else delete f["n-cpu-moe"];
    delete f["cpu-moe"];
  } else {
    f["fit"] = sugg.fit.fit;
    f["fitc"] = String(sugg.fit.fitc);
    f["fitt"] = String(sugg.fit.fitt);
    f["ctk"] = sugg.fit.ctk;
    f["ctv"] = sugg.fit.ctv;
    delete f["ngl"]; delete f["c"];
    // pin expert offload (fit only auto-tunes unset args); always n-cpu-moe
    if ("n-cpu-moe" in sugg.fit) f["n-cpu-moe"] = String(sugg.fit["n-cpu-moe"]);
    else delete f["n-cpu-moe"];
    delete f["cpu-moe"];
  }
  current.suggestion_snapshot = sugg;
  renderEditor(current);
  toast(`applied ${kind} suggestion`, "ok");
}

async function rememberComboValues() {
  if (!state.settings) return;
  const presets = state.settings.value_presets || (state.settings.value_presets = {});
  let changed = false;
  for (const k of comboKeys()) {
    const v = (current.flags[k] || "").trim();
    if (!v) continue;
    const arr = presets[k] || (presets[k] = []);
    if (!arr.includes(v)) { arr.push(v); changed = true; }
  }
  if (changed) {
    try { state.settings = await api.saveSettings(state.settings); } catch { /* non-fatal */ }
  }
}

async function save() {
  if (!current) return;
  try {
    await rememberComboValues();
    await api.updateConfig(current.id, {
      name: current.name,
      model_display_path: current.model_display_path,
      flags: current.flags,
      mmproj: current.mmproj,
      notes: current.notes,
      suggestion_snapshot: current.suggestion_snapshot || null,
    });
    const { loadConfigs } = await import("./config_list.js");
    await loadConfigs(true);
    toast("saved", "ok");
  } catch (e) {
    toast(`save failed: ${e.message}`, "error");
  }
}

async function del() {
  if (!current || !confirm(`Delete config "${current.name}"?`)) return;
  try {
    await api.deleteConfig(current.id);
    state.selectedConfigId = null;
    const { loadConfigs } = await import("./config_list.js");
    await loadConfigs(false);
    toast("deleted", "ok");
  } catch (e) {
    toast(`delete failed: ${e.message}`, "error");
  }
}
