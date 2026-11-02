// Monitor tab: polls /v1/status every few seconds while it is open.
"use strict";
(() => {
  const REFRESH_MS = 3000;
  let timer = null;

  const gib = (bytes) => (bytes / 2 ** 30).toFixed(2) + " GiB";
  const rate = (v) => (v > 0 ? v.toFixed(1) : "–");
  function duration(seconds) {
    if (seconds < 60) return Math.round(seconds) + " s";
    if (seconds < 3600) return Math.floor(seconds / 60) + " min";
    return Math.floor(seconds / 3600) + " h " + Math.floor((seconds % 3600) / 60) + " min";
  }
  function card(title, big, sub, fraction) {
    const div = document.createElement("div");
    div.className = "card";
    const h = document.createElement("h3");
    h.textContent = title;
    const b = document.createElement("div");
    b.className = "big";
    b.textContent = big;
    div.append(h, b);
    if (sub) {
      const s = document.createElement("div");
      s.className = "sub";
      s.textContent = sub;
      div.append(s);
    }
    if (fraction !== undefined) {
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("div");
      fill.style.width = Math.min(100, Math.max(0, fraction * 100)).toFixed(1) + "%";
      bar.append(fill);
      div.append(bar);
    }
    return div;
  }

  function render(s) {
    const m = s.memory, t = s.totals;
    const state = s.active
      ? "Generating"
      : s.queued ? "Queued" : "Idle";
    const stateSub = s.active
      ? s.active.generated_tokens + " tokens in " + duration(s.active.seconds) + (s.queued ? " · " + s.queued + " waiting" : "")
      : s.queued + " waiting";
    const last = s.recent[0];
    $("cards").replaceChildren(
      card("Loaded model", s.model ? titles[s.model] || s.model : "None", s.context ? "context " + s.context.toLocaleString() : "loads with the first request"),
      card("Engine", state, stateSub),
      card("Metal memory", gib(m.metal_allocated_bytes),
        "peak " + gib(m.metal_peak_bytes) + " of " + gib(m.recommended_working_set_bytes) + " working set",
        m.recommended_working_set_bytes ? m.metal_allocated_bytes / m.recommended_working_set_bytes : 0),
      card("Last request", last ? rate(last.decode_tokens_per_second) + " tok/s" : "–",
        last ? "prefill " + rate(last.prefill_tokens_per_second) + " tok/s" + (last.mtp ? " · MTP" : "") : "no requests yet"),
      card("Kept sessions", s.sessions.kept + " / " + s.sessions.capacity, "for prompt prefix reuse"),
      card("Totals", t.requests + " requests",
        t.prompt_tokens + " prompt (" + t.reused_tokens + " reused) · " + t.completion_tokens + " generated · up " + duration(s.uptime_seconds)),
    );
    const table = document.createElement("table");
    const columns = [["Finished", true], ["Model", true], ["Prompt", false], ["Reused", false], ["Output", false],
      ["Prefill tok/s", false], ["Decode tok/s", false], ["Finish", true], ["MTP", true]];
    const head = table.insertRow();
    for (const [name, left] of columns) {
      const th = document.createElement("th");
      th.textContent = name;
      if (left) th.className = "left";
      head.append(th);
    }
    for (const r of s.recent) {
      const row = table.insertRow();
      const values = [new Date(r.finished * 1000).toLocaleTimeString(), r.model, r.prompt_tokens, r.reused_tokens,
        r.completion_tokens, rate(r.prefill_tokens_per_second), rate(r.decode_tokens_per_second), r.finish, r.mtp ? "on" : "off"];
      values.forEach((v, i) => {
        const cell = row.insertCell();
        cell.textContent = v;
        if (columns[i][1]) cell.className = "left";
      });
    }
    const title = document.createElement("h3");
    title.className = "sub";
    title.textContent = s.recent.length ? "Recent requests" : "No requests yet. Speeds of finished requests appear here.";
    $("recent").replaceChildren(title, ...(s.recent.length ? [table] : []));
  }

  async function refresh() {
    clearTimeout(timer);
    try {
      const response = await api("/v1/status", {}, false);
      if (response.status === 401) {
        const button = document.createElement("button");
        button.textContent = "Enter API key";
        button.onclick = async () => { await api("/v1/status"); refresh(); };
        $("cards").replaceChildren(card("Monitor", "API key required", "this server was started with --api-key"));
        $("recent").replaceChildren(button);
        return;
      }
      if (!response.ok) throw new Error("HTTP " + response.status);
      render(await response.json());
    } catch (e) {
      $("cards").replaceChildren(card("Monitor", "Server unreachable", e.message || String(e)));
    }
    if (document.body.dataset.view === "monitor" && !document.hidden) timer = setTimeout(refresh, REFRESH_MS);
  }

  window.addEventListener("ingot-view", (e) => {
    if (e.detail === "monitor") refresh();
    else clearTimeout(timer);
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && document.body.dataset.view === "monitor") refresh();
  });
})();
