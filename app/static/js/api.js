// Thin fetch wrappers around the backend API.
async function req(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const ct = res.headers.get("content-type") || "";
  const payload = ct.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = payload && payload.detail ? payload.detail : payload;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export const api = {
  // models
  listModels: () => req("GET", "/api/models"),
  rescan: () => req("POST", "/api/models/scan"),
  metadata: (path) => req("GET", `/api/models/metadata?path=${encodeURIComponent(path)}`),
  // flags
  flags: () => req("GET", "/api/flags"),
  refreshFlags: () => req("POST", "/api/flags/refresh"),
  // configs
  configs: () => req("GET", "/api/configs"),
  createConfig: (b) => req("POST", "/api/configs", b),
  updateConfig: (id, b) => req("PUT", `/api/configs/${id}`, b),
  deleteConfig: (id) => req("DELETE", `/api/configs/${id}`),
  preview: () => req("GET", "/api/configs/preview"),
  export: (b) => req("POST", "/api/configs/export", b || {}),
  import: (b) => req("POST", "/api/configs/import", b || {}),
  // hardware / suggest
  hardware: () => req("GET", "/api/hardware"),
  suggest: (b) => req("POST", "/api/suggest", b),
  // settings
  settings: () => req("GET", "/api/settings"),
  saveSettings: (b) => req("PUT", "/api/settings", b),
};
