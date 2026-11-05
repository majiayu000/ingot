// The frame around the views: greeting, starters, the sidebar toggle and the
// engine line at the foot of the sidebar.
"use strict";
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
        if (!s.model) show("No model loaded", "Loads with your first message", false);
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
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { greet(); refresh(); } });
  refresh();
})();
