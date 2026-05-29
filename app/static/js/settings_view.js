import { api } from "./api.js";
import { state, el, toast } from "./state.js";

export async function loadSettings() {
  state.settings = await api.settings();
  renderSettings();
}

function renderSettings() {
  const form = document.getElementById("settings-form");
  const s = state.settings;
  form.innerHTML = "";

  // scan roots
  const rootsList = el("div", { class: "roots-list" });
  const addRoot = (v = "") => rootsList.appendChild(rootRow(v, rootsList));
  (s.scan_roots.length ? s.scan_roots : [""]).forEach(addRoot);
  form.appendChild(el("div", { class: "fld" },
    el("label", {}, "scan roots (recursively scanned for *.gguf)"),
    rootsList,
    el("button", { class: "btn small", onclick: () => addRoot("") }, "+ add root"),
  ));

  const exe = textField("llama-server exe (for flag scanning)", s.llama_server_exe);
  const out = textField("output INI path", s.output_ini_path);
  const headroom = numField("VRAM headroom fraction (0–1)", s.headroom_frac, "0.01");
  const reserve = numField("compute reserve (MiB)", s.compute_reserve_mib, "1");
  const manual = numField("manual VRAM (MiB, optional)", s.manual_vram_mib ?? "", "1");
  form.append(exe.wrap, out.wrap, headroom.wrap, reserve.wrap, manual.wrap);

  form.appendChild(el("button", { class: "btn primary", onclick: save }, "Save settings"));

  async function save() {
    const roots = [...rootsList.querySelectorAll("input")].map((i) => i.value.trim()).filter(Boolean);
    const body = {
      scan_roots: roots,
      llama_server_exe: exe.input.value.trim(),
      output_ini_path: out.input.value.trim(),
      headroom_frac: Number(headroom.input.value) || 0.1,
      compute_reserve_mib: Number(reserve.input.value) || 1024,
      manual_vram_mib: manual.input.value ? Number(manual.input.value) : null,
      value_presets: (state.settings && state.settings.value_presets) || {},
    };
    try {
      state.settings = await api.saveSettings(body);
      toast("settings saved", "ok");
    } catch (e) {
      toast(`save failed: ${e.message}`, "error");
    }
  }
}

function rootRow(value, list) {
  const input = el("input", { value });
  return el("div", { class: "root-row" }, input,
    el("button", { class: "btn small danger", onclick: (e) => e.target.closest(".root-row").remove() }, "✕"));
}
function textField(label, value) {
  const input = el("input", { value: value || "" });
  return { input, wrap: el("div", { class: "fld" }, el("label", {}, label), input) };
}
function numField(label, value, step) {
  const input = el("input", { type: "number", step: step || "any", value: value === null ? "" : value });
  return { input, wrap: el("div", { class: "fld" }, el("label", {}, label), input) };
}
