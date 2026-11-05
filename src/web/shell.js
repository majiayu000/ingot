// The frame around the views: greeting, starters, the sidebar toggle and the
// engine line at the foot of the sidebar.
"use strict";
// Appearance: "system" (no attribute), "light" or "dark"; set from Settings.
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
applyTheme(load("theme", "system"));

// Answer versions from Retry: m.versions holds the others and m.at is the
// shown one's place among all of them.
function snapshot(m) { return { content: m.content, reasoning: m.reasoning || "", meta: m.meta || "", format: m.format }; }
function allVersions(m) {
  const others = m.versions || [];
  const at = m.at === undefined ? others.length : m.at;
  return [...others.slice(0, at), snapshot(m), ...others.slice(at)];
}
function versionSwitcher(m) {
  const all = allVersions(m), at = m.at ?? m.versions.length;
  const show = (place) => {
    const list = allVersions(m);
    Object.assign(m, list[place]);
    m.versions = list.filter((_, i) => i !== place);
    m.at = place;
    persist();
    render();
  };
  const box = element("span", "versions");
  const back = action("‹", "Previous answer", () => show(at - 1));
  const next = action("›", "Next answer", () => show(at + 1));
  back.disabled = at === 0;
  next.disabled = at === all.length - 1;
  box.append(back, element("span", "", `${at + 1} / ${all.length}`), next);
  return box;
}

// A new chat holding the conversation up to message `index`. From a
// question, the question goes back into the composer to be edited and sent.
function branchFrom(index) {
  if (controller || !current) return;
  const source = current.messages[index];
  const c = newChat();
  c.title = ("Branch: " + current.title).slice(0, 80);
  c.system = current.system;
  c.project = current.project || "";
  c.messages = structuredClone(current.messages.slice(0, source.role === "user" ? index : index + 1));
  chats.push(c);
  current = c;
  persist();
  show();
  if (source.role === "user") { $("input").value = source.content; fitInput(); }
  $("input").focus();
}

// Finished downloads are announced when Ingot is not in front.
function askToNotify() {
  if (!window.webkit && "Notification" in window && Notification.permission === "default") Notification.requestPermission();
}
function notify(title, body) {
  if (document.hasFocus() && !document.hidden) return;
  if (native({ action: "notify", title, body })) return;
  if ("Notification" in window && Notification.permission === "granted") new Notification(title, { body });
}

// Opening a search result marks the matches and scrolls to the first.
function highlightMatches(text) {
  const needle = text.toLowerCase();
  const walker = document.createTreeWalker($("log"), NodeFilter.SHOW_TEXT);
  const hits = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node.nodeValue.toLowerCase().includes(needle)) hits.push(node);
  }
  let first = null;
  for (const node of hits) {
    const parts = node.nodeValue.split(new RegExp(`(${text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "i"));
    const marked = parts.map((part, i) => {
      if (i % 2 === 0) return document.createTextNode(part);
      const mark = element("mark", "", part);
      first = first || mark;
      return mark;
    });
    node.replaceWith(...marked);
  }
  if (first) first.scrollIntoView({ block: "center" });
}

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
  function showEngine(model, state, on) {
    $("engine-model").textContent = model;
    $("engine-state").textContent = state;
    $("engine-dot").classList.toggle("on", on);
  }
  // While the engine is down the notice offers a retry; when it is back the
  // model list, which failed at start, loads again.
  let down = false;
  function engineDown(on) {
    $("offline").hidden = !on;
    if (down && !on) {
      $("status").textContent = "";
      loadModels().catch(() => null);
    }
    down = on;
  }
  $("offline-retry").onclick = () => { if (!native({ action: "restart" })) refresh(); };
  $("first-pick").onclick = () => go("models");
  async function refresh() {
    clearTimeout(timer);
    try {
      const response = await api("/v1/status", {}, false);
      if (response.status === 401) showEngine("Ingot", "API key needed", false);
      else if (!response.ok) throw new Error("HTTP " + response.status);
      else {
        const s = await response.json();
        const m = s.memory;
        // Friendly names come from the catalog, which may not be loaded yet.
        if (s.model && !titles[s.model] && !catalog) await catalogReady().catch(() => null);
        if (!s.model) showEngine("No model loaded", "Loads when you chat", false);
        else {
          const memory = m.recommended_working_set_bytes
            ? `${gib(m.metal_allocated_bytes)} of ${gib(m.recommended_working_set_bytes)} GiB`
            : `${gib(m.metal_allocated_bytes)} GiB`;
          showEngine(titles[s.model] || s.model, s.active ? "Generating" : "Loaded, " + memory, true);
        }
      }
      engineDown(false);
    } catch (e) {
      showEngine("Ingot", "Engine not running", false);
      engineDown(true);
    }
    if (!document.hidden) timer = setTimeout(refresh, REFRESH_MS);
  }
  // Starred chats stay at the top of the list.
  $("star").onclick = () => {
    if (!current) return;
    current.starred = !current.starred;
    persist();
    show();
  };
  // One conversation as Markdown: the app opens a save panel, a browser downloads it.
  $("export").onclick = () => {
    if (!current) return;
    const parts = [`# ${current.title}`];
    if (current.system.trim()) parts.push(`> System: ${current.system.trim()}`);
    for (const m of current.messages) {
      if (m.role === "user") parts.push("## You", m.content);
      else if (m.role === "assistant") parts.push("## Ingot", m.content);
    }
    const text = parts.join("\n\n") + "\n";
    const name = (current.title.replace(/[\\/:*?"<>|]+/g, " ").trim().slice(0, 60) || "conversation") + ".md";
    if (native({ action: "export", text, name })) return;
    const link = element("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
    link.download = name;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  // Saved prompts live in the server settings, next to the chats.
  let prompts = null;
  async function savePrompts(list) {
    try {
      prompts = (await json("/api/settings", { method: "PUT", body: JSON.stringify({ prompts: list }) })).prompts;
    } catch (e) {
      $("status").textContent = "Could not save prompts: " + e.message;
    }
    renderPrompts();
  }
  function renderPrompts() {
    const box = $("prompts");
    const items = (prompts || []).map((text, i) => {
      const row = element("div", "prompt");
      const use = element("button", "plain", text.split("\n")[0].slice(0, 80));
      use.title = text.slice(0, 400);
      use.onclick = () => {
        const input = $("input");
        input.value = input.value.trim() ? input.value.replace(/\s*$/, "\n\n") + text : text;
        closePrompts();
        fitInput();
        input.focus();
      };
      const del = action("✕", "Remove this prompt", () => savePrompts(prompts.filter((_, j) => j !== i)));
      row.append(use, del);
      return row;
    });
    const add = element("button", "plain add", "Save current text as a prompt");
    add.disabled = !$("input").value.trim();
    add.onclick = () => savePrompts([...(prompts || []), $("input").value.trim()]);
    box.replaceChildren(...(items.length ? items : [element("div", "note", "No saved prompts yet.")]), add);
  }
  function closePrompts() {
    $("prompts").hidden = true;
    $("prompts-toggle").setAttribute("aria-expanded", "false");
  }
  $("prompts-toggle").onclick = async (e) => {
    e.stopPropagation();
    if (!$("prompts").hidden) return closePrompts();
    if (prompts === null) prompts = (await json("/api/settings").catch(() => ({}))).prompts || [];
    renderPrompts();
    $("prompts").hidden = false;
    $("prompts-toggle").setAttribute("aria-expanded", "true");
  };
  document.addEventListener("click", (e) => { if (!$("prompts").hidden && !$("prompts").contains(e.target)) closePrompts(); });

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
    } else if (e.key === "Escape" && !$("prompts").hidden) {
      closePrompts();
    } else if (e.key === "Escape" && !$("stop").hidden) {
      $("stop").click();
    }
  });

  document.addEventListener("visibilitychange", () => { if (!document.hidden) { greet(); refresh(); } });
  refresh();
})();

// Startup runs last, once every script has loaded: the chain below calls into
// models.js, and the restored page needs its script listening for "ingot-view".
readChats().then(() => {
  show();
  return catalogReady();
}).then(() => loadModels()).then((ids) => {
  // A new Mac starts on Models: there is nothing to chat with yet.
  if (ids && !ids.length) go("models");
}).catch((e) => {
  $("model").replaceChildren(new Option("No model", ""));
  $("status").textContent = "Cannot list models (" + e.message + "); is the server running?";
});
// The app restarts the engine when server settings change: return to the page.
const lastView = load("view", "chat");
go(["model", "custom"].includes(lastView) ? "models" : lastView);
$("input").focus();
