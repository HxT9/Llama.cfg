// Light/dark theme toggle, persisted in localStorage.
const KEY = "llamacfg-theme";

function apply(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = theme === "dark" ? "☀️" : "🌙";
}

export function initTheme() {
  apply(localStorage.getItem(KEY) || "light");
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    apply(next);
    localStorage.setItem(KEY, next);
  });
}
