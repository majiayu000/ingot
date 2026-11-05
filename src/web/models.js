// Models: what this Mac can run, Hugging Face search, one model's details,
// custom models, downloads and deletion.
"use strict";
const GIB = 1024 ** 3;
const size = (bytes) => (bytes / GIB).toFixed(bytes < 10 * GIB ? 1 : 0) + " GiB";
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(label, onclick, kind) {
  const node = element("button", kind || "", label);
  node.onclick = onclick;
  return node;
}
function field(label, control) {
  const wrap = element("label", "field", label);
  wrap.append(control);
  return wrap;
}
async function failureOf(response) {
  const body = await response.json().catch(() => ({}));
  return body.error && body.error.message ? body.error.message : body.error || "HTTP " + response.status;
}
async function json(path, options) {
  const response = await api(path, options);
  if (!response.ok) throw new Error(await failureOf(response));
  return response.json();
}
const NOTES = {
  memory: (m) => [`Runs in memory: ${m.memory_gib} GiB, context ${m.context}`, ""],
  streaming: () => ["Part of the model stays on SSD, so answers are slower", "slow"],
  too_large: () => ["Too large for this Mac", "warn"],
  no_disk: () => ["Not enough free disk space", "warn"],
  unknown: () => ["", ""],
  custom: () => ["", ""],
};
let catalog = null, libraryError = "", confirming = "", tab = "installed";
let results = null, resultFiles = {}, searching = false, supportedOnly = true, query = "";
// Downloads this page runs: model name to its abort controller and progress.
const running = new Map();

async function catalogReady() {
  catalog = await json("/api/catalog");
  for (const model of catalog.models) titles[model.name] = model.title;
  return catalog;
}
async function refreshLibrary() {
  try {
    await catalogReady();
  } catch (e) {
    libraryError = "Cannot read the model list: " + e.message;
  }
  renderLibrary();
}

async function pull(name, total) {
  const controller = new AbortController();
  const state = { controller, files: {}, started: Date.now(), done: 0, resumed: null, total };
  askToNotify();
  running.set(name, state);
  libraryError = "";
  renderLibrary();
  try {
    const response = await api("/api/pull", { method: "POST", body: JSON.stringify({ model: name }), signal: controller.signal });
    if (!response.ok) throw new Error(await failureOf(response));
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      const lines = text.split("\n");
      text = lines.pop();
      for (const line of lines) {
        const event = JSON.parse(line);
        if (event.error) throw new Error(event.error);
        if (event.digest) {
          state.files[event.digest] = event;
          state.done = Object.values(state.files).reduce((sum, f) => sum + f.completed, 0);
          if (state.resumed === null) state.resumed = state.done;
        }
      }
      showProgress(name);
    }
    notify(`${titles[name] || name} is ready`, "The download finished. You can chat with it now.");
  } catch (e) {
    if (e.name !== "AbortError") {
      libraryError = `${titles[name] || name}: ${e.message}`;
      notify(`${titles[name] || name} did not download`, e.message);
    }
  }
  running.delete(name);
  await refreshLibrary();
  await loadModels().catch(() => null);
}

function showProgress(name) {
  const state = running.get(name);
  const node = document.querySelector(`[data-progress="${CSS.escape(name)}"]`);
  if (!state || !node) return;
  const total = Math.max(state.total, Object.values(state.files).reduce((sum, f) => sum + f.total, 0));
  const seconds = (Date.now() - state.started) / 1000;
  const speed = (state.done - (state.resumed || 0)) / Math.max(seconds, 1);
  const left = speed > 0 ? (total - state.done) / speed : 0;
  const time = left > 90 ? Math.round(left / 60) + " min left" : Math.round(left) + " s left";
  node.querySelector(".bar div").style.width = (100 * state.done / Math.max(total, 1)) + "%";
  node.querySelector("span").textContent = state.done
    ? `${size(state.done)} of ${size(total)}` + (speed > 0 ? ` · ${(speed / 1e6).toFixed(0)} MB/s · ${time}` : "")
    : "Starting…";
}
function progressLine(name) {
  const line = element("div", "progress");
  line.dataset.progress = name;
  const bar = element("div", "bar");
  bar.append(element("div"));
  line.append(element("span", "", "Starting…"), bar);
  return line;
}

async function removeModel(name) {
  confirming = "";
  const response = await api("/api/delete", { method: "DELETE", body: JSON.stringify({ model: name }) });
  libraryError = response.ok ? "" : `${titles[name] || name}: ${await failureOf(response)}`;
  await refreshLibrary();
  await loadModels().catch(() => null);
}
// An empty Ollama generate request with keep_alive 0 unloads the model.
async function unload(name, target) {
  target.disabled = true;
  const response = await api("/api/generate", { method: "POST", body: JSON.stringify({ model: name, keep_alive: 0 }) });
  if (!response.ok) libraryError = `${titles[name] || name}: ${await failureOf(response)}`;
  await refreshLibrary();
  openModel(name);
}
function chatWith(name) {
  $("model").value = name;
  $("model").dispatchEvent(new Event("change"));
  go("chat");
  $("input").focus();
}

function actions(model) {
  const box = element("div", "buttons");
  const state = running.get(model.name);
  const blocked = !model.present && ["too_large", "no_disk"].includes(model.fit);
  if (state) box.append(button("Stop", () => state.controller.abort()));
  else if (model.downloading) box.append(element("span", "note", "Downloading…"));
  else if (model.present) box.append(button("Chat", () => chatWith(model.name)));
  else {
    const download = button("Download " + size(model.download_bytes), () => pull(model.name, model.download_bytes), model.recommended ? "gold" : "");
    download.disabled = blocked;
    box.append(download);
  }
  return box;
}
function card(model, tile, badge) {
  const blocked = !model.present && ["too_large", "no_disk"].includes(model.fit);
  const node = element("div", "model" + (tile ? " tile" : "") + (tile && model.recommended ? " best" : ""));
  const body = element("div", "body");
  if (badge) body.append(element("span", "badge" + (model.recommended ? " gold" : ""), badge));
  body.append(button(model.title, () => openModel(model.name), "name"));
  if (model.summary) body.append(element("div", "about", model.summary));
  const [note, tone] = NOTES[model.fit](model);
  if (note) body.append(element("div", "note " + (blocked ? "warn" : tone), note));
  // A custom model runs on its base model's weights, so it is ready when they are loaded.
  if (model.loaded) body.append(element("div", "note good", model.custom ? `Ready, shares ${titles[model.base] || model.base}'s memory` : "Loaded"));
  node.append(body, actions(model));
  if (running.has(model.name)) node.append(progressLine(model.name));
  return node;
}
function segmented() {
  const box = element("div", "segmented");
  for (const [id, label] of [["installed", "Library"], ["discover", "Discover"]]) {
    const b = button(label, () => { tab = id; renderLibrary(); });
    b.setAttribute("aria-pressed", String(tab === id));
    box.append(b);
  }
  return box;
}
function installedPage(nodes) {
  const none = !catalog.models.some((m) => m.present);
  const head = element("div", "head");
  const text = element("div");
  text.append(element("h1", "", none ? "Choose your first model" : "Models"),
    element("p", "lead", `${catalog.mac.device}: ${catalog.mac.memory_gib} GiB of memory for models, ${size(catalog.mac.free_bytes)} free on disk`));
  head.append(text, button("New custom model", () => openCustom(), "dark"), segmented());
  nodes.push(head);
  if (libraryError) nodes.push(element("div", "error-line", libraryError));
  const best = catalog.models.find((m) => m.recommended);
  const small = none && catalog.models.filter((m) => m.fit === "memory" && !m.recommended)
    .sort((a, b) => a.download_bytes - b.download_bytes)[0];
  let rest = catalog.models;
  if (none && best) {
    const hero = element("div", "hero");
    hero.append(card(best, true, "Best for this Mac"));
    if (small) hero.append(card(small, true, "Fastest to try"));
    nodes.push(hero);
    rest = rest.filter((m) => m !== best && m !== small);
  }
  const badge = (m) => card(m, false, m.recommended ? "Best for this Mac" : "");
  for (const [label, models] of [["On this Mac", rest.filter((m) => m.present)], ["Available to download", rest.filter((m) => !m.present)]]) {
    if (!models.length) continue;
    const list = element("div", "list");
    list.append(...models.map(badge));
    nodes.push(element("h2", "group", label), list);
  }
}

async function search(text) {
  query = text;
  searching = true;
  resultFiles = {};
  libraryError = "";
  renderLibrary();
  try {
    results = (await json("/api/search?q=" + encodeURIComponent(text))).results;
  } catch (e) {
    results = [];
    libraryError = "Search failed: " + e.message;
  }
  searching = false;
  renderLibrary();
  $("hf-query").focus();
}
async function openRepository(id) {
  if (resultFiles[id]) { delete resultFiles[id]; return renderLibrary(); }
  try {
    resultFiles[id] = (await json("/api/search/files?repo=" + encodeURIComponent(id))).files;
  } catch (e) {
    libraryError = `${id}: ${e.message}`;
  }
  renderLibrary();
}
function fileRow(repo, file) {
  const label = file.file.split("/").pop();
  const row = element("div", "setting");
  const text = element("div", "", label);
  const notes = { memory: `Estimated ${file.memory_gib} GiB of memory`, too_large: "Too large for this Mac", no_disk: "Not enough free disk space" };
  text.append(element("small", "", size(file.bytes) + ". " + notes[file.fit]));
  row.append(text);
  if (running.has(file.name)) {
    row.append(button("Stop", () => running.get(file.name).controller.abort()));
  } else if (file.present) {
    row.append(button("Chat", () => chatWith(file.name), "gold"));
  } else {
    const download = button("Download", async () => { await pull(file.name, file.bytes); delete resultFiles[repo]; openRepository(repo); }, "gold");
    download.disabled = file.fit !== "memory";
    row.append(download);
  }
  const wrap = element("div");
  wrap.append(row);
  if (running.has(file.name)) wrap.append(progressLine(file.name));
  return wrap;
}
function discoverPage(nodes) {
  const head = element("div", "head");
  const text = element("div");
  text.append(element("h1", "", "Discover models"), element("p", "lead", "Search Hugging Face. Ingot checks each file before you download it."));
  head.append(text, segmented());
  nodes.push(head);
  const form = document.createElement("form");
  form.className = "searchbox surface";
  const input = element("input");
  input.id = "hf-query";
  input.placeholder = "Search GGUF models, for example qwen or gemma";
  input.setAttribute("aria-label", "Search models");
  input.value = query;
  form.append(input);
  form.onsubmit = (e) => { e.preventDefault(); if (input.value.trim()) search(input.value.trim()); };
  nodes.push(form);
  const only = button("Runs on Ingot", () => { supportedOnly = !supportedOnly; renderLibrary(); });
  only.setAttribute("aria-pressed", String(supportedOnly));
  const filters = element("div", "buttons");
  filters.append(only);
  nodes.push(filters);
  if (libraryError) nodes.push(element("div", "error-line", libraryError));
  if (searching) nodes.push(element("div", "note", "Searching…"));
  if (!results) return;
  const list = element("div", "list");
  const shown = results.filter((r) => !supportedOnly || r.supported);
  if (!shown.length && !searching) nodes.push(element("div", "note", "No matching models."));
  for (const r of shown) {
    const node = element("div", "result");
    const line = element("div", "head");
    const info = element("div");
    info.append(element("div", "", r.id),
      element("div", "note " + (r.supported ? "good" : "warn"), (r.architecture || "Unknown architecture") + (r.supported ? ". Runs on Ingot" : ". Not supported yet")));
    line.append(info);
    const open = button(resultFiles[r.id] ? "Hide files" : "Choose file", () => openRepository(r.id));
    open.disabled = !r.supported;
    line.append(open);
    node.append(line);
    if (resultFiles[r.id]) {
      if (!resultFiles[r.id].length) node.append(element("div", "note", "No GGUF files in this repository."));
      node.append(...resultFiles[r.id].map((f) => fileRow(r.id, f)));
    }
    list.append(node);
  }
  nodes.push(list);
}
function renderLibrary() {
  const nodes = [];
  if (!catalog) nodes.push(element("div", "error-line", libraryError));
  else if (tab === "installed") installedPage(nodes);
  else discoverPage(nodes);
  $("library").replaceChildren(...nodes);
  for (const name of running.keys()) showProgress(name);
}

function backButton() {
  const row = element("div", "back");
  row.append(button("‹ Models", () => go("models"), "plain"));
  return row;
}
function stat(label, value, note) {
  const node = element("div", "card stat");
  node.append(element("small", "", label), element("b", "", value), element("small", "", note));
  return node;
}
function facts(pairs) {
  const box = element("div", "facts");
  for (const [label, value] of pairs) box.append(element("span", "", label), value instanceof Node ? value : element("span", "", value));
  return box;
}
async function openModel(name) {
  go("model");
  const page = $("detail");
  page.replaceChildren(backButton());
  let m;
  try {
    m = await json("/api/model?name=" + encodeURIComponent(name));
  } catch (e) {
    page.append(element("div", "error-line", e.message));
    return;
  }
  const head = element("div", "head");
  const text = element("div");
  text.append(element("h1", "", m.title), element("p", "lead", m.custom ? `Custom model based on ${m.base}` : m.summary || ""));
  head.append(text);
  if (m.present) head.append(button("Customize", () => openCustom(m.custom || { base: m.base })), button("Chat", () => chatWith(name), "gold"));
  const files = m.files || [];
  const total = files.reduce((sum, f) => sum + f.bytes, 0);
  const modes = { memory: "Every part in memory", streaming: "Part of the model on SSD", too_large: "Too large for this Mac", no_disk: "Not enough free disk space", unknown: "Not rated" };
  const stats = element("div", "stats");
  stats.append(stat("Memory", m.memory_gib ? m.memory_gib + " GiB" : "Not measured", m.cap_gib ? `of ${m.cap_gib} GiB here` : ""),
    stat("Download", size(total), m.present ? "Installed" : "Not installed"),
    stat("Context", m.context ? m.context.toLocaleString() : "Automatic", "Tokens"),
    stat("Mode", m.loaded ? "Loaded" : "Idle", modes[m.fit] || ""));
  const here = element("div", "panel");
  const list = [["Tool calling", m.tools ? "Supported" : "Not supported"], ["Image input", m.vision ? "Ready" : "Not available"]];
  if (m.draft) list.push(["Faster decoding", "MTP draft"]);
  here.append(element("h2", "", "Capabilities"), facts(list));
  if (m.loaded) {
    const line = element("div", "buttons");
    line.append(button("Unload", (e) => unload(name, e.target)), element("span", "note", "Frees its memory now. Unload it to delete it."));
    here.append(line);
  } else if (m.present) {
    // Deleting needs a second click that names what goes.
    const label = m.custom ? "Delete custom model" : "Delete from disk";
    const del = button(label, async () => {
      if (confirming !== name) {
        confirming = name;
        del.textContent = m.custom ? "Click again to delete" : `Click again to delete ${size(total)}`;
        return;
      }
      await removeModel(name);
      go("models");
    }, "danger");
    confirming = "";
    const line = element("div", "buttons");
    line.append(del);
    here.append(line);
  }
  const panel = element("div", "panel");
  panel.append(element("h2", "", "Files"), facts(files.map((f) => [f.name, size(f.bytes)])));
  if (m.source) {
    const link = element("a", "", "Model page and license");
    link.href = m.source;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    panel.append(link);
  }
  const split = element("div", "split");
  split.append(here, panel);
  // Replace rather than append: a second open may have finished first.
  page.replaceChildren(backButton(), head, stats, split);
  if (m.custom) {
    const box = element("div", "panel");
    box.append(element("h2", "", "System prompt"), element("div", "note", m.custom.system || "None"));
    page.append(box);
  }
}

const SLIDERS = [
  ["num_ctx", "Context", 1024, 65536, 1024, 8192],
  ["temperature", "Temperature", 0, 2, 0.05, 0],
  ["top_p", "Top P", 0.05, 1, 0.05, 1],
  ["top_k", "Top K", 0, 200, 1, 0],
  ["min_p", "Min P", 0, 0.5, 0.01, 0],
  ["repeat_penalty", "Repeat penalty", 1, 2, 0.05, 1],
];
function openCustom(existing) {
  go("custom");
  const start = existing || {};
  const parameters = Object.assign({}, start.parameters);
  const page = $("custom");
  const head = element("div", "head");
  const text = element("div");
  text.append(element("h1", "", start.name ? "Edit custom model" : "New custom model"), element("p", "lead", "Saves your prompt and settings under a new name. No weights are copied."));
  const name = element("input");
  name.value = start.name || "";
  name.placeholder = "rust-reviewer";
  const base = element("select");
  const bases = catalog.models.filter((m) => m.present && !m.custom);
  base.append(...bases.map((m) => new Option(m.title, m.name)));
  if (start.base) base.value = start.base;
  const system = element("textarea");
  system.value = start.system || "";
  system.placeholder = "How this model should behave";
  const stops = element("input");
  stops.value = (parameters.stop || []).join(", ");
  stops.placeholder = "Comma separated";
  const message = element("div", "error-line");
  const sliders = element("div", "panel");
  sliders.append(element("h2", "", "Parameters"));
  const changed = {};
  for (const [key, label, min, max, step, fallback] of SLIDERS) {
    const row = element("label", "slider");
    const value = element("span", "", String(parameters[key] ?? fallback));
    const head2 = element("div");
    head2.append(element("span", "", label), value);
    const input = element("input");
    Object.assign(input, { type: "range", min, max, step, value: parameters[key] ?? fallback });
    input.oninput = () => { value.textContent = input.value; changed[key] = Number(input.value); };
    row.append(head2, input);
    sliders.append(row);
  }
  async function submit() {
    const merged = Object.assign({}, parameters, changed);
    const stop = stops.value.split(",").map((s) => s.trim()).filter(Boolean);
    if (stop.length) merged.stop = stop; else delete merged.stop;
    try {
      await json("/api/create", { method: "POST", body: JSON.stringify({ model: name.value.trim(), from: base.value, system: system.value, parameters: merged }) });
      await refreshLibrary();
      await loadModels();
      openModel(name.value.trim());
    } catch (e) {
      message.textContent = e.message;
    }
  }
  const file = element("input");
  file.type = "file";
  file.hidden = true;
  file.onchange = async () => {
    const chosen = file.files[0];
    if (!chosen) return;
    const modelfile = await chosen.text();
    const target = name.value.trim() || chosen.name.replace(/\.[^.]*$/, "");
    try {
      await json("/api/create", { method: "POST", body: JSON.stringify({ model: target, modelfile }) });
      await refreshLibrary();
      await loadModels();
      openModel(target);
    } catch (e) {
      message.textContent = e.message;
    }
  };
  head.append(text, button("Import Modelfile", () => file.click()), button("Save model", submit, "gold"));
  const left = element("div", "panel");
  left.append(field("Name", name), field("Based on", base), field("System prompt", system), field("Stop sequences", stops));
  const split = element("div", "split");
  split.append(left, sliders);
  page.replaceChildren(backButton(), head, message, split, file);
}

window.addEventListener("ingot-view", (e) => { if (e.detail === "models") refreshLibrary(); });
