// Chat: conversations, projects, files and streaming replies.
"use strict";
const $ = (id) => document.getElementById(id);
const STORE = "ingot.chat.v1";

// Storage can be missing or throw (private windows, blocked site data); the
// page then works for this visit only and says so.
let storageWarned = false;
function load(key, fallback) {
  try {
    const raw = localStorage.getItem(STORE + "." + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (e) {
    return fallback;
  }
}
function save(key, value, quiet) {
  try {
    localStorage.setItem(STORE + "." + key, JSON.stringify(value));
    return true;
  } catch (e) {
    if (quiet) return false;
    if (!storageWarned) {
      storageWarned = true;
      $("status").textContent = "Conversations cannot be saved in this browser (" + (e.name || "storage error") + "); they last until the page is closed.";
    }
  }
}

// Asked in the page: embedded browsers may not support window.prompt.
function askKey(text) {
  return new Promise((resolve) => {
    $("keylabel").textContent = text;
    $("keyinput").value = "";
    $("keybox").hidden = false;
    $("keyinput").focus();
    const done = (value) => { $("keybox").hidden = true; resolve(value); };
    $("keyform").onsubmit = (e) => { e.preventDefault(); done($("keyinput").value.trim()); };
    $("keycancel").onclick = () => done("");
  });
}
// A one-line question asked in the page; "" when cancelled.
function askText(text) {
  return new Promise((resolve) => {
    $("keylabel").textContent = text;
    $("keyinput").type = "text";
    $("keyinput").value = "";
    $("keybox").hidden = false;
    $("keyinput").focus();
    const done = (value) => { $("keybox").hidden = true; $("keyinput").type = "password"; resolve(value); };
    $("keyform").onsubmit = (e) => { e.preventDefault(); done($("keyinput").value); };
    $("keycancel").onclick = () => done("");
  });
}

// With `serve --api-key` every API call needs the key: ask once, keep it.
async function api(path, options = {}, ask = true) {
  for (let attempt = 0; ; attempt++) {
    const key = load("apiKey", "");
    const headers = Object.assign({}, options.headers);
    const secret = window.ingotKey || key;
    if (secret) headers.Authorization = "Bearer " + secret;
    const response = await fetch(path, Object.assign({}, options, { headers }));
    if (response.status !== 401 || !ask || attempt >= 2) return response;
    const entered = await askKey(key ? "The API key was not accepted. API key:" : "This server requires an API key:");
    if (!entered) return response;
    save("apiKey", entered);
  }
}

// chats: [{id, title, system, messages: [{role, content, images?, reasoning?, meta?, format?}], updated}]
// images are data URLs; when storage is full they are kept for this visit only.
// Display names of models, filled from the catalog.
const titles = {};
let chats = [];
let current = null;
let controller = null;
// The app keeps chats in a file on this Mac. A browser on another device is
// refused that file and falls back to its own storage.
let chatsOnDisk = true;
let saveTimer = null;
let projectFilter = null;
let searchText = "";
let contextSize = 0;

function withoutImages(c) {
  return { ...c, messages: c.messages.map((m) => m.images ? { ...m, images: undefined, imagesDropped: m.images.length } : m) };
}
async function writeChats() {
  const stored = chats.map(withoutImages);
  if (!chatsOnDisk) { save("chats", stored); return; }
  const response = await api("/api/chats", { method: "PUT", body: JSON.stringify(stored) }).catch(() => null);
  if (!response || !response.ok) $("status").textContent = "Conversations could not be saved" + (response ? " (HTTP " + response.status + ")." : ".");
}
function persist() {
  save("current", current ? current.id : null);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(writeChats, 300);
}
async function readChats() {
  const response = await api("/api/chats").catch(() => null);
  if (response && response.ok) {
    chats = await response.json();
    const old = load("chats", []);
    // Chats from before they moved to a file.
    if (!chats.length && Array.isArray(old) && old.length) { chats = old; persist(); }
  } else {
    chatsOnDisk = false;
    chats = load("chats", []);
    if (!Array.isArray(chats)) chats = [];
  }
  current = chats.find((c) => c.id === load("current", null)) || null;
}
function newChat() {
  return { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), title: "New chat", system: "", project: projectFilter || "", messages: [], updated: Date.now() };
}
function chat() {
  if (!current) {
    current = newChat();
    current.system = $("system").value;
    chats.unshift(current);
  }
  return current;
}

// ---------- Markdown (everything is HTML-escaped before tags are added) ----------
function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
// Inline Markdown on already escaped text; only http(s) links are allowed.
function inline(text) {
  const codes = [];
  text = text.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  text = text
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)"]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  return text.replace(/\u0000(\d+)\u0000/g, (_, i) => "<code>" + codes[i] + "</code>");
}
const cells = (row) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
function markdown(source) {
  const lines = escapeHtml(source).split("\n");
  const out = [];
  let paragraph = [], list = null;
  const flush = () => {
    if (paragraph.length) { out.push("<p>" + paragraph.map(inline).join("<br>") + "</p>"); paragraph = []; }
    if (list) { out.push("<" + list.tag + ">" + list.items.map((i) => "<li>" + inline(i) + "</li>").join("") + "</" + list.tag + ">"); list = null; }
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]*)/);
    if (fence) {
      flush();
      const code = [];
      while (++i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i]);
      // The language is a word of [\w+#.-], so it is safe as text.
      out.push('<div class="codeblock"><div class="codebar"><span>' + (fence[1] || "code") +
        '</span><button class="icon copy-code">Copy</button></div><pre><code>' + code.join("\n") + "</code></pre></div>");
      continue;
    }
    let m;
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      flush();
      const head = cells(line), rows = [];
      i++;
      while (i + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[i + 1])) rows.push(cells(lines[++i]));
      out.push("<table><tr>" + head.map((c) => "<th>" + inline(c) + "</th>").join("") + "</tr>" +
        rows.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("") + "</table>");
    } else if ((m = line.match(/^(#{1,4})\s+(.*)$/))) {
      flush();
      out.push("<h" + m[1].length + ">" + inline(m[2]) + "</h" + m[1].length + ">");
    } else if ((m = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/))) {
      const tag = /\d/.test(m[1]) ? "ol" : "ul";
      if (paragraph.length || (list && list.tag !== tag)) flush();
      if (!list) list = { tag, items: [] };
      list.items.push(m[2]);
    } else if ((m = line.match(/^&gt;\s?(.*)$/))) {
      flush();
      out.push("<blockquote>" + inline(m[1]) + "</blockquote>");
    } else if (!line.trim()) {
      flush();
    } else {
      if (list) flush();
      paragraph.push(line);
    }
  }
  flush();
  return out.join("");
}

// ---------- Rendering ----------
// The answer follows new text only while the reader is at the bottom.
let following = true;
function scrollDown(force) {
  const s = $("scroll");
  if (force) following = true;
  if (following) s.scrollTop = s.scrollHeight;
  $("latest").hidden = following;
}
$("scroll").addEventListener("scroll", () => {
  const s = $("scroll");
  following = s.scrollHeight - s.scrollTop - s.clientHeight < 48;
  $("latest").hidden = following;
});
$("latest").onclick = () => scrollDown(true);

async function copy(text, button) {
  const label = button.textContent;
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch (e) {
    // No async clipboard (plain http on a LAN address) or permission denied.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.cssText = "position:fixed;opacity:0";
    document.body.append(area);
    area.select();
    try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
    area.remove();
  }
  button.textContent = ok ? "Copied" : "Copy failed";
  setTimeout(() => { button.textContent = label; }, 1200);
}
function action(label, title, onclick) {
  const b = document.createElement("button");
  b.className = "icon";
  b.textContent = label;
  b.title = title;
  b.onclick = onclick;
  return b;
}
function formatted(m) {
  if (m.format !== "json") return m.content;
  try { return "```json\n" + JSON.stringify(JSON.parse(m.content), null, 2) + "\n```"; } catch (e) { return m.content; }
}

function bubble(m, index, last) {
  const div = document.createElement("div");
  div.className = "msg " + m.role;
  if (m.role === "error") {
    div.textContent = m.content;
    return div;
  }
  const actions = document.createElement("div");
  actions.className = "actions";
  if (m.role === "user") {
    if ((m.images && m.images.length) || m.imagesDropped) {
      const thumbs = document.createElement("div");
      thumbs.className = "thumbs";
      for (const url of m.images || []) {
        const img = document.createElement("img");
        img.src = url;
        img.alt = "Attached image";
        thumbs.append(img);
      }
      if (!m.images && m.imagesDropped) {
        const missing = document.createElement("span");
        missing.className = "missing";
        missing.textContent = m.imagesDropped + " image(s) not saved in this browser";
        thumbs.append(missing);
      }
      div.append(thumbs);
    }
    if (m.files && m.files.length) {
      const list = document.createElement("div");
      list.className = "chips";
      for (const f of m.files) list.append(fileChip(f));
      div.append(list);
    }
    const text = document.createElement("div");
    text.className = "text";
    text.textContent = m.content;
    div.append(text);
    if (index === last.user && !controller) actions.append(action("Edit", "Edit this message and generate again", () => edit(index)));
  } else {
    if (m.reasoning) {
      const d = document.createElement("details");
      d.className = "thinking";
      const s = document.createElement("summary");
      s.textContent = m.streaming && !m.content ? "Thinking…" : "Thought process";
      const body = document.createElement("div");
      body.textContent = m.reasoning;
      d.append(s, body);
      div.append(d);
    }
    const body = document.createElement("div");
    body.className = "md" + (m.streaming ? " cursor" : "");
    body.innerHTML = markdown(formatted(m));
    for (const button of body.querySelectorAll(".copy-code")) {
      button.onclick = () => copy(button.closest(".codeblock").querySelector("code").textContent, button);
    }
    div.append(body);
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = m.meta || "";
    actions.append(meta);
    if (index === last.assistant && !controller) actions.append(action("Retry", "Generate this answer again", regenerate));
  }
  if (!m.streaming) {
    const button = action("Copy", "Copy this message", () => copy(m.content, button));
    actions.append(button);
  }
  div.append(actions);
  return div;
}

function matchesSearch(c) {
  if (!searchText) return true;
  const text = searchText.toLowerCase();
  return c.title.toLowerCase().includes(text) ||
    c.messages.some((m) => m.content.toLowerCase().includes(text));
}
function projects() {
  const counts = new Map();
  for (const c of chats) if (c.project) counts.set(c.project, (counts.get(c.project) || 0) + 1);
  return [...counts].sort((a, b) => a[0].localeCompare(b[0]));
}
function renderProjects() {
  const ul = $("projects");
  ul.replaceChildren(...projects().map(([name, count]) => {
    const li = document.createElement("li");
    li.className = name === projectFilter ? "active" : "";
    const label = document.createElement("span");
    label.textContent = name;
    const number = document.createElement("small");
    number.textContent = count;
    li.append(label, number);
    li.onclick = () => { projectFilter = projectFilter === name ? null : name; renderChats(); };
    return li;
  }));
  $("projects-title").hidden = !ul.children.length;
  const select = $("project");
  const value = current ? current.project || "" : projectFilter || "";
  select.replaceChildren(new Option("No project", ""),
    ...projects().map(([name]) => new Option(name, name)), new Option("New project…", "\u0001"));
  select.value = value;
}
function renderChats() {
  renderProjects();
  const ul = $("chats");
  ul.replaceChildren();
  const list = chats.filter((c) => (!projectFilter || c.project === projectFilter) && matchesSearch(c))
    .sort((a, b) => b.updated - a.updated);
  $("chats-title").textContent = searchText ? "Results" : projectFilter || "Recents";
  for (const c of list) {
    const li = document.createElement("li");
    li.className = c === current ? "active" : "";
    const title = document.createElement("span");
    title.textContent = c.title;
    title.title = c.title;
    const rename = action("Rename", "Rename this conversation", (e) => { e.stopPropagation(); renameChat(c, title); });
    title.ondblclick = rename.onclick;
    const del = action("✕", "Delete this conversation", (e) => {
      e.stopPropagation();
      if (controller) return;
      chats = chats.filter((x) => x !== c);
      if (current === c) current = null;
      persist();
      show();
    });
    li.append(title, rename, del);
    li.onclick = () => {
      if (controller) return;
      current = c;
      persist();
      document.body.classList.remove("sidebar");
      go("chat");
      show();
    };
    ul.append(li);
  }
}

function render() {
  const log = $("log");
  log.replaceChildren();
  const messages = current ? current.messages : [];
  document.body.classList.toggle("blank", !messages.length);
  $("input").placeholder = messages.length ? "Reply to Ingot…" : "How can I help you today?";
  const last = { user: -1, assistant: -1 };
  messages.forEach((m, i) => {
    if (m.role === "user") last.user = i;
    if (m.role === "assistant") last.assistant = i;
  });
  messages.forEach((m, i) => log.append(bubble(m, i, last)));
  scrollDown(true);
}

function show() {
  $("system").value = current ? current.system : "";
  $("chat-title").textContent = current && current.messages.length ? current.title : "";
  renderChats();
  render();
  setBusy(!!controller);
}

function setBusy(busy) {
  $("send").hidden = busy;
  $("stop").hidden = !busy;
  $("new").disabled = busy;
  $("model").disabled = busy;
  $("thinking").disabled = busy;
  updateAttach();
}

// ---------- Generation ----------
function errorMessage(value) {
  return (value && value.error && value.error.message) || JSON.stringify(value);
}

async function generate() {
  const c = chat();
  c.messages = c.messages.filter((m) => m.role !== "error");
  const messages = [];
  if (c.system.trim()) messages.push({ role: "system", content: c.system.trim() });
  for (const m of c.messages) {
    const parts = (m.images || []).map((url) => ({ type: "image_url", image_url: { url } }));
    const documents = (m.files || []).map((f) => '<file name="' + f.name + '">\n' + f.text + "\n</file>\n\n").join("");
    const text = documents + m.content;
    const message = { role: m.role, content: parts.length ? [...parts, { type: "text", text }] : text };
    if (m.reasoning) message.reasoning_content = m.reasoning;
    messages.push(message);
  }
  const body = { messages, stream: true, stream_options: { include_usage: true } };
  if ($("model").value) body.model = $("model").value;
  const temperature = $("temperature").value, maxTokens = $("max_tokens").value;
  if (temperature !== "") body.temperature = Number(temperature);
  if (maxTokens !== "") body.max_tokens = Number(maxTokens);
  const format = $("format").value;
  if (format === "json") body.response_format = { type: "json_object" };
  if ($("thinking").value) body.reasoning_effort = $("thinking").value;

  const reply = { role: "assistant", content: "", reasoning: "", meta: "", streaming: true, format };
  c.messages.push(reply);
  c.updated = Date.now();
  controller = new AbortController();
  setBusy(true);
  renderChats();
  render();
  let node = $("log").lastElementChild;
  const refresh = () => { const fresh = bubble(reply, -1, {}); node.replaceWith(fresh); node = fresh; scrollDown(); };

  $("status").textContent = "Loading the model if needed. Large models take a minute…";
  const start = performance.now();
  let first = null, finish = null, usage = null, failed = null, last = start, pieces = 0;
  try {
    const response = await api("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail;
      try { detail = errorMessage(await response.json()); } catch (e) { detail = response.statusText; }
      throw new Error("HTTP " + response.status + ": " + detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", pending = false;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of event.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (chunk.error) { failed = errorMessage(chunk); continue; }
          if (chunk.usage) usage = chunk.usage;
          const choice = chunk.choices && chunk.choices[0];
          if (!choice) continue;
          const delta = choice.delta || {};
          if (delta.content || delta.reasoning_content) {
            if (first === null) first = performance.now();
            last = performance.now();
            reply.content += delta.content || "";
            reply.reasoning += delta.reasoning_content || "";
            pieces++;
            pending = true;
          }
          if (choice.finish_reason) finish = choice.finish_reason;
        }
      }
      if (pending) {
        pending = false;
        refresh();
        const seconds = (performance.now() - first) / 1000;
        $("status").textContent = "Generating…" + (seconds > 0.5 ? " " + ((pieces - 1) / seconds).toFixed(1) + " tokens/s" : "");
      }
    }
    if (failed) throw new Error(failed);
  } catch (e) {
    if (e.name === "AbortError") finish = "stopped";
    else failed = e.message || String(e);
  } finally {
    controller = null;
  }
  const end = finish === "stopped" ? last : performance.now();
  const parts = [];
  if (usage) parts.push(usage.prompt_tokens + " prompt + " + usage.completion_tokens + " completion tokens");
  const decodeSeconds = first === null ? 0 : (end - first) / 1000;
  // Without usage (stopped early) each streamed chunk is about one token.
  const generated = usage ? usage.completion_tokens : pieces;
  if (generated > 1 && decodeSeconds > 0) parts.push((usage ? "" : "≈") + ((generated - 1) / decodeSeconds).toFixed(1) + " tokens/s");
  if (first !== null) parts.push(((first - start) / 1000).toFixed(1) + " s to first token");
  if (finish) parts.push(finish === "length" ? "stopped at max tokens" : finish);
  reply.meta = parts.join(" · ");
  delete reply.streaming;
  if (usage) showContext(usage.prompt_tokens + usage.completion_tokens);
  if (!reply.content && !reply.reasoning) c.messages.pop();
  if (failed) c.messages.push({ role: "error", content: failed });
  if (c.title === "New chat") {
    const firstUser = c.messages.find((m) => m.role === "user");
    if (firstUser) c.title = firstUser.content.trim().split("\n")[0].slice(0, 60) || (firstUser.images ? "Image" : "New chat");
  }
  persist();
  $("status").textContent = storageWarned ? $("status").textContent : "";
  show();
}

function send() {
  const text = $("input").value;
  if ((!text.trim() && !attachments.length && !documents.length) || controller) return;
  $("input").value = "";
  fitInput();
  const c = chat();
  c.messages = c.messages.filter((m) => m.role !== "error");
  const m = { role: "user", content: text };
  if (attachments.length) m.images = attachments;
  if (documents.length) m.files = documents;
  attachments = [];
  documents = [];
  renderAttachments();
  c.messages.push(m);
  generate();
}

// ---------- Files ----------
// Images for the next message (data URLs), documents as extracted text, and
// which models can read images.
let attachments = [];
let documents = [];
let vision = {};
const MAX_TEXT = 4 * 1024 * 1024;
function canSee() { return !!vision[$("model").value]; }
function updateAttach() { $("attach").disabled = !!controller; }
function fileChip(f) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = f.name;
  if (f.pages) {
    const pages = document.createElement("small");
    pages.textContent = f.pages + (f.pages === 1 ? " page" : " pages");
    chip.append(pages);
  }
  return chip;
}
function renderAttachments() {
  const box = $("attachments");
  box.replaceChildren(...attachments.map((url, i) => {
    const t = document.createElement("div");
    t.className = "thumb";
    const img = document.createElement("img");
    img.src = url;
    img.alt = "Image to send";
    t.append(img, action("✕", "Remove this image", () => { attachments.splice(i, 1); renderAttachments(); }));
    return t;
  }), ...documents.map((f, i) => {
    const chip = fileChip(f);
    chip.append(action("✕", "Remove this file", () => { documents.splice(i, 1); renderAttachments(); }));
    return chip;
  }));
}
function readURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}
// PNG and JPEG are sent as is; other image types are converted to PNG.
async function addFile(file) {
  if (!file) return;
  if (file.type.startsWith("image/")) return addImage(file);
  try {
    if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) {
      const response = await api("/api/extract", { method: "POST", body: file });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || "HTTP " + response.status);
      documents.push({ name: file.name, pages: value.pages, text: value.text });
    } else {
      if (file.size > MAX_TEXT) throw new Error("the file is larger than 4 MiB");
      const text = await file.text();
      if (text.includes("\u0000")) throw new Error("this is not a text file");
      documents.push({ name: file.name, text });
    }
    renderAttachments();
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Cannot read " + file.name + ": " + (e.message || e);
  }
}
async function addImage(file) {
  if (!canSee()) { $("status").textContent = "This model cannot read images. Qwen3.8 with its vision projector can."; return; }
  try {
    let url;
    if (file.type === "image/png" || file.type === "image/jpeg") url = await readURL(file);
    else {
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      url = canvas.toDataURL("image/png");
    }
    attachments.push(url);
    renderAttachments();
  } catch (e) {
    $("status").textContent = "Cannot read the image (" + (e.message || e) + ").";
  }
}
$("attach").onclick = () => $("file").click();
$("file").onchange = () => { for (const f of $("file").files) addFile(f); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  const files = [...(e.clipboardData ? e.clipboardData.items : [])].filter((i) => i.kind === "file").map((i) => i.getAsFile());
  if (!files.length) return;
  e.preventDefault();
  files.forEach(addFile);
});
const footer = document.querySelector("footer");
footer.addEventListener("dragover", (e) => { e.preventDefault(); footer.classList.add("dragging"); });
footer.addEventListener("dragleave", () => footer.classList.remove("dragging"));
footer.addEventListener("drop", (e) => {
  e.preventDefault();
  footer.classList.remove("dragging");
  for (const f of e.dataTransfer.files) addFile(f);
});

function edit(index) {
  if (controller || !current) return;
  $("input").value = current.messages[index].content;
  attachments = current.messages[index].images || [];
  documents = current.messages[index].files || [];
  renderAttachments();
  current.messages = current.messages.slice(0, index);
  persist();
  show();
  $("input").focus();
}

$("send").onclick = send;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
$("stop").onclick = () => { if (controller) controller.abort(); };
$("new").onclick = () => {
  if (controller) return;
  current = null;
  persist();
  document.body.classList.remove("sidebar");
  go("chat");
  show();
  $("input").focus();
};
function regenerate() {
  if (controller || !current) return;
  const m = current.messages;
  while (m.length && m[m.length - 1].role !== "user") m.pop();
  if (m.length) generate();
}
$("options-toggle").onclick = () => {
  const box = $("options");
  box.hidden = !box.hidden;
  $("options-toggle").setAttribute("aria-expanded", String(!box.hidden));
};
function fitInput() {
  const box = $("input");
  box.style.height = "auto";
  box.style.height = box.scrollHeight + "px";
}
$("input").addEventListener("input", fitInput);
$("system").addEventListener("input", () => {
  if (!current) return;
  current.system = $("system").value;
  persist();
});
$("menu").onclick = () => document.body.classList.toggle("sidebar");
$("scroll").addEventListener("click", () => document.body.classList.remove("sidebar"));

// Settings that persist across conversations.
for (const id of ["temperature", "max_tokens", "format", "thinking"]) {
  $(id).value = load(id, "");
  $(id).addEventListener("change", () => save(id, $(id).value));
}
$("model").addEventListener("change", () => { save("model", $("model").value); updateAttach(); });
$("search").addEventListener("input", () => { searchText = $("search").value.trim(); renderChats(); });
$("project").addEventListener("change", async () => {
  let name = $("project").value;
  if (name === "\u0001") {
    name = (await askText("Name of the new project:")).trim();
    if (!name) return renderChats();
  }
  chat().project = name;
  persist();
  renderChats();
});

// Fills the model menu; returns the local model ids.
function loadModels() {
  return api("/v1/models").then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }).then(showModels);
}
function showModels(v) {
  const ids = (v.data || []).map((m) => m.id).filter(Boolean);
  if (!ids.length) {
    $("model").replaceChildren(new Option("No model yet", ""));
    updateAttach();
    return ids;
  }
  vision = Object.fromEntries((v.data || []).map((m) => [m.id, !!m.vision]));
  const select = $("model");
  select.replaceChildren(...ids.map((id) => new Option((titles[id] || id) + (vision[id] ? " · vision" : ""), id)));
  const saved = load("model", "");
  if (ids.includes(saved)) select.value = saved;
  document.title = "Ingot";
  updateAttach();
  return ids;
}
function showContext(used) {
  const meter = $("context");
  api("/v1/status", {}, false).then((r) => r.json()).then((s) => {
    contextSize = s.context || contextSize;
    meter.hidden = !contextSize;
    meter.hidden = used < contextSize / 2;
    $("context-text").textContent = Math.round(100 * used / Math.max(contextSize, 1)) + "% of context used";
  }).catch(() => { meter.hidden = true; });
}

// ---------- Views ----------
function go(name) {
  document.body.dataset.view = name;
  save("view", name);
  for (const button of document.querySelectorAll("[data-go]")) {
    const on = button.dataset.go === name || (button.dataset.go === "models" && ["model", "custom"].includes(name));
    button.setAttribute("aria-pressed", String(on));
  }
  window.dispatchEvent(new CustomEvent("ingot-view", { detail: name }));
}
for (const button of document.querySelectorAll("[data-go]")) button.onclick = () => go(button.dataset.go);
// Asks the Mac app to do what a page cannot; false in a plain browser.
function native(message) {
  const handler = window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.ingot;
  if (!handler) return false;
  handler.postMessage(message);
  return true;
}
if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.ingot) document.body.classList.add("native");

const modelsReady = readChats().then(() => {
  show();
  return catalogReady();
}).then(() => loadModels()).catch((e) => {
  $("model").replaceChildren(new Option("No model", ""));
  $("status").textContent = "Cannot list models (" + e.message + "); is the server running?";
  return null;
});
