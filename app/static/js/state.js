// Shared in-memory state + tiny helpers.
export const state = {
  models: [],
  modelsByPath: {},
  flags: [],
  flagsByGroup: {},
  configs: [],
  selectedConfigId: null,
  hardware: null,
  settings: null,
};

export function indexModels() {
  state.modelsByPath = {};
  for (const m of state.models) state.modelsByPath[m.display_path] = m;
}

export function indexFlags() {
  state.flagsByGroup = {};
  for (const f of state.flags) {
    const g = f.group || "other";
    (state.flagsByGroup[g] = state.flagsByGroup[g] || []).push(f);
  }
}

// DOM helpers
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

export function fmtBytes(n) {
  if (!n) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

let toastTimer = null;
export function toast(msg, kind = "") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), 3500);
}
