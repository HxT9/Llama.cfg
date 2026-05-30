import { api } from "./api.js";
import { state, indexFlags, toast } from "./state.js";
import { initModelsView, loadModels, setCreateHandler } from "./models_view.js";
import { initConfigList, loadConfigs, createFromModel } from "./config_list.js";
import { loadSettings } from "./settings_view.js";
import { initTheme } from "./theme.js";

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((t) => t.addEventListener("click", () => {
    tabs.forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    document.getElementById("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "preview") refreshPreview();
  }));
}

async function loadFlags() {
  try {
    state.flags = await api.flags();
    indexFlags();
  } catch (e) {
    toast(`flag scan failed: ${e.message}`, "error");
  }
}

async function loadHardware() {
  try {
    state.hardware = await api.hardware();
    const hw = state.hardware;
    const gpu = hw.gpus[0];
    const badge = document.getElementById("hw-badge");
    badge.textContent = gpu
      ? `GPU ${gpu.name || ""} ${gpu.free_mib}/${gpu.total_mib} MiB · RAM ${Math.round(hw.ram_available_mib / 1024)}/${Math.round(hw.ram_total_mib / 1024)} GB`
      : `no GPU (manual) · RAM ${Math.round(hw.ram_available_mib / 1024)} GB`;
  } catch {
    document.getElementById("hw-badge").textContent = "hw: n/a";
  }
}

async function refreshPreview() {
  const pre = document.getElementById("ini-preview");
  const msg = document.getElementById("preview-msg");
  msg.textContent = "";
  try {
    pre.textContent = await api.preview();
  } catch (e) {
    pre.textContent = "";
    msg.textContent = e.message;
  }
}

function initPreview() {
  document.getElementById("btn-refresh-preview").addEventListener("click", refreshPreview);
  document.getElementById("btn-export").addEventListener("click", async () => {
    const msg = document.getElementById("preview-msg");
    try {
      const r = await api.export();
      toast(`exported ${r.bytes} bytes → ${r.path}`, "ok");
      refreshPreview();
    } catch (e) { msg.textContent = e.message; toast(`export failed: ${e.message}`, "error"); }
  });
  document.getElementById("btn-import").addEventListener("click", async () => {
    if (!confirm("Import from the output INI file? This replaces the current config list.")) return;
    try {
      await api.import({ replace: true });
      await loadConfigs(false);
      toast("imported", "ok");
    } catch (e) { toast(`import failed: ${e.message}`, "error"); }
  });
}

async function main() {
  initTheme();
  initTabs();
  initModelsView();
  initConfigList();
  initPreview();
  setCreateHandler(createFromModel);
  await Promise.all([loadHardware(), loadFlags(), loadSettings()]);
  await Promise.all([loadModels(false), loadConfigs(false)]);
}

main();
