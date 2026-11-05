// The frame around the views: greeting, starters, the sidebar toggle and the
// engine line at the foot of the sidebar.
"use strict";
// Appearance: "system" (no attribute), "light" or "dark"; set from Settings.
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
applyTheme(load("theme", "system"));

// Called from the conversation list in chat.js.
function renameChat(c, title) {
  const input = document.createElement("input");
  input.value = c.title;
  input.setAttribute("aria-label", "Conversation name");
  let done = false;
  const finish = (keep) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (keep && name) { c.title = name.slice(0, 80); persist(); }
    show();
  };
  input.onclick = (e) => e.stopPropagation();
  input.onkeydown = (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
  input.onblur = () => finish(true);
  title.replaceWith(input);
  input.select();
}
(() => {
  const REFRESH_MS = 5000;
  let timer = null;

  function greet() {
    const hour = new Date().getHours();
    $("greeting").textContent = hour < 5 ? "Good evening" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
  }
  greet();

  for (const button of document.querySelectorAll("[data-start]")) {
    button.onclick = () => {
      const start = button.dataset.start;
      if (start === "pdf") return $("attach").click();
      if (start === "models") return go("models");
      $("input").value = start;
      fitInput();
      $("input").focus();
    };
  }

  function collapse(on) {
    document.body.classList.toggle("collapsed", on);
    save("collapsed", on, true);
  }
  $("collapse").onclick = () => collapse(true);
  $("expand").onclick = () => collapse(false);
  collapse(load("collapsed", false) === true);

  const gib = (bytes) => Math.round(bytes / 2 ** 30);
  function show(model, state, on) {
    $("engine-model").textContent = model;
    $("engine-state").textContent = state;
    $("engine-dot").classList.toggle("on", on);
  }
  async function refresh() {
    clearTimeout(timer);
    try {
      const response = await api("/v1/status", {}, false);
      if (response.status === 401) show("Ingot", "API key needed", false);
      else if (!response.ok) throw new Error("HTTP " + response.status);
      else {
        const s = await response.json();
        const m = s.memory;
        if (!s.model) show("No model loaded", "Loads when you chat", false);
        else {
          const memory = m.recommended_working_set_bytes
            ? `${gib(m.metal_allocated_bytes)} of ${gib(m.recommended_working_set_bytes)} GiB`
            : `${gib(m.metal_allocated_bytes)} GiB`;
          show(titles[s.model] || s.model, s.active ? "Generating" : "Loaded, " + memory, true);
        }
      }
    } catch (e) {
      show("Ingot", "Engine not running", false);
    }
    if (!document.hidden) timer = setTimeout(refresh, REFRESH_MS);
  }
  // Command-N new chat, Command-K search, Escape stops an answer.
  document.addEventListener("keydown", (e) => {
    const command = e.metaKey || e.ctrlKey;
    if (command && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "n") {
      e.preventDefault();
      $("new").click();
    } else if (command && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
      e.preventDefault();
      collapse(false);
      $("search").focus();
      $("search").select();
    } else if (e.key === "Escape" && !$("stop").hidden) {
      $("stop").click();
    }
  });

  document.addEventListener("visibilitychange", () => { if (!document.hidden) { greet(); refresh(); } });
  refresh();
})();

// The app restarts the engine when server settings change: return to the page.
// This runs last, once every page script listens for "ingot-view".
const lastView = load("view", "chat");
go(["model", "custom"].includes(lastView) ? "models" : lastView);
$("input").focus();
