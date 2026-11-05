// Settings and Connect apps. The server's own settings take effect when it
// restarts: the Mac app does that itself, a terminal user does it by hand.
"use strict";
(() => {
  let settings = null, models = [], message = "", update = null;

  function switchButton(key) {
    const node = element("button", "switch");
    node.setAttribute("role", "switch");
    node.setAttribute("aria-checked", String(!!settings[key]));
    node.onclick = () => change({ [key]: !settings[key] });
    return node;
  }
  function row(label, note, control) {
    const node = element("div", "setting");
    const text = element("div", "", label);
    if (note) text.append(element("small", "", note));
    if (control.matches("select, input, .switch")) control.setAttribute("aria-label", label);
    node.append(text, control);
    return node;
  }
  function picker(key, options) {
    const select = element("select");
    select.setAttribute("aria-label", key);
    select.append(...options.map(([value, label]) => new Option(label, value)));
    select.value = String(settings[key]);
    select.onchange = () => change({ [key]: key === "context" ? Number(select.value) : select.value });
    return select;
  }
  const APP_ONLY = ["check_updates", "menu_bar"];
  async function change(values) {
    try {
      settings = await json("/api/settings", { method: "PUT", body: JSON.stringify(values) });
      native({ action: "settings" });
      if (!Object.keys(values).every((key) => APP_ONLY.includes(key))) {
        message = native({ action: "restart" }) ? "" : "Saved. Restart `ingot serve` to apply server settings.";
      }
    } catch (e) {
      message = e.message;
    }
    render();
  }
  function randomKey() {
    const bytes = crypto.getRandomValues(new Uint8Array(24));
    return "ingot-" + [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  function exportChats() {
    const text = JSON.stringify(chats, null, 2);
    if (native({ action: "export", text })) return;
    const link = element("a");
    link.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    link.download = "ingot-chats.json";
    link.click();
    URL.revokeObjectURL(link.href);
  }
  let confirmingDelete = false, section = "General";
  // Open at login belongs to the Mac app; a browser has no such setting.
  function loginSwitch() {
    const node = element("button", "switch");
    node.setAttribute("role", "switch");
    node.setAttribute("aria-checked", String(!!window.ingotLogin));
    node.onclick = () => native({ action: "login", on: !window.ingotLogin });
    return node;
  }
  window.ingotLoginChanged = (on, error) => {
    window.ingotLogin = on;
    message = error ? "Open at login is not available: " + error : "";
    render();
  };
  function render() {
    const page = $("preferences");
    if (!settings) return;
    const folder = button("Change…", () => {
      if (!native({ action: "chooseFolder" })) message = "Set INGOT_MODELS or use the Mac app to change the folder.", render();
    });
    const contexts = [[0, "Automatic for this Mac"], [4096, "4,096"], [8192, "8,192"], [16384, "16,384"], [32768, "32,768"], [65536, "65,536"]];
    const theme = element("select");
    theme.append(new Option("Match system", "system"), new Option("Light", "light"), new Option("Dark", "dark"));
    theme.value = load("theme", "system");
    theme.onchange = () => { save("theme", theme.value, true); applyTheme(theme.value); };
    const general = [element("h2", "", "General"),
      row("Appearance", "", theme),
      row("Default model", "Used for new chats and connected apps", picker("default_model", [["", "Most recent"], ...models.map((m) => [m, titles[m] || m])])),
      row("Context length", "Longer context needs more memory", picker("context", contexts)),
      row("Unload after idle", "Frees memory when you stop chatting", picker("keep_alive", [["5m", "5 minutes"], ["10m", "10 minutes"], ["1h", "1 hour"], ["-1", "Never"]])),
      row("Model folder", settings.models_dir || (catalog ? catalog.directory : ""), folder),
      row("Keep running in the menu bar", "When off, closing the window quits Ingot", switchButton("menu_bar"))];
    if (typeof window.ingotLogin === "boolean") general.push(row("Open at login", "", loginSwitch()));
    // The Mac app installs updates itself; a browser can only point at them.
    let updates = switchButton("check_updates");
    updates.setAttribute("aria-label", "Check for updates automatically");
    if (document.body.classList.contains("native")) {
      const line = element("div", "buttons");
      line.append(button("Check now", () => native({ action: "checkUpdates" })), updates);
      updates = line;
    }
    general.push(row("Check for updates automatically", update || "", updates));
    const server = element("div", "panel");
    const port = element("input");
    port.type = "number";
    port.value = settings.port;
    port.setAttribute("aria-label", "Port");
    port.onchange = () => change({ port: Number(port.value) });
    const key = element("div", "buttons");
    if (settings.api_key) {
      key.append(button("Copy", (e) => copy(settings.api_key, e.target)), button("Remove", () => change({ api_key: "", allow_network: false })));
    } else {
      key.append(button("Create key", () => change({ api_key: randomKey() })));
    }
    const network = switchButton("allow_network");
    if (!settings.api_key) network.onclick = async () => change({ api_key: randomKey(), allow_network: true });
    const address = button("Copy", (e) => copy(location.origin, e.target));
    server.append(element("h2", "", "Server"), row("Address", location.origin, address), row("Port", "", port),
      row("Allow other devices", "Needs an API key. Traffic is not encrypted.", network),
      row("API key", settings.api_key ? "Set" : "Not set", key));
    const clear = button(confirmingDelete ? `Delete ${chats.length} chats` : "Delete all chats", () => {
      if (!confirmingDelete) { confirmingDelete = true; return render(); }
      confirmingDelete = false;
      chats = [];
      current = null;
      persist();
      show();
      render();
    }, "danger");
    const line = element("div", "buttons");
    line.append(button("Export chats", exportChats), clear);
    const privacy = [element("h2", "", "Privacy"),
      element("div", "note", "Conversations stay on this Mac. Ingot connects to the internet only to search and download models, and to check for updates."), line];
    const shown = { General: general, Server: [...server.children], Privacy: privacy }[section];
    const nav = element("nav");
    nav.setAttribute("aria-label", "Settings sections");
    for (const name of ["General", "Server", "Connect apps", "Monitor", "Privacy"]) {
      const target = { "Connect apps": "connect", Monitor: "monitor" }[name];
      const item = button(name, () => { if (target) return go(target); section = name; message = ""; render(); }, "plain");
      if (name === section) item.setAttribute("aria-current", "page");
      nav.append(item);
    }
    const body = element("div");
    body.append(...shown);
    const prefs = element("div", "prefs");
    prefs.append(nav, body);
    page.replaceChildren(element("h1", "", "Settings"), element("div", "error-line", message), prefs);
  }
  async function openSettings() {
    try {
      settings = await json("/api/settings");
      // The model folder shown when none is set comes from the catalog.
      if (!settings.models_dir && !catalog) await catalogReady().catch(() => null);
      models = (await json("/v1/models")).data.map((m) => m.id);
      if (settings.check_updates) {
        json("/api/update").then((u) => {
          update = u.available ? `Version ${u.latest} is available` : `Version ${u.current} is up to date`;
          render();
        }).catch(() => { update = "Could not check for updates"; render(); });
      }
    } catch (e) {
      $("preferences").replaceChildren(element("h1", "", "Settings"), element("div", "error-line", "Settings are available only on this Mac. " + e.message));
      return;
    }
    render();
  }
  // The Mac app answers a folder choice here.
  window.ingotFolderChosen = (path) => change({ models_dir: path });

  async function openConnect() {
    const page = $("connect");
    const head = element("div");
    head.append(element("h1", "", "Connect your apps"), element("p", "lead", "Use your local models in the tools you already work with."));
    const nodes = [head];
    let found;
    try {
      found = await json("/api/apps");
      models = (await json("/v1/models")).data.map((m) => m.id);
    } catch (e) {
      page.replaceChildren(head, element("div", "error-line", e.message));
      return;
    }
    const url = location.origin;
    const server = element("div", "panel");
    const address = element("div", "address");
    address.append(element("code", "", url), button("Copy", (e) => copy(url, e.target)));
    const select = element("select");
    select.append(...models.map((m) => new Option(titles[m] || m, m)));
    select.value = $("model").value;
    server.append(element("div", "note good", "Ready"), address, field("Model for coding tools", select));
    nodes.push(server);
    const help = element("div", "error-line");
    const list = element("div", "apps");
    for (const app of found.apps) {
      const card = element("div", "model");
      const body = element("div", "body");
      body.append(element("div", "", app.name), element("div", "note", app.installed ? "Installed" : "Not found on this Mac"));
      const action = app.launch
        ? button("Launch", (e) => launchTool(app, select.value, e.target))
        : button("Show steps", () => { help.textContent = app.steps; help.style.color = "var(--text)"; });
      if (app.launch) action.disabled = !app.installed;
      card.append(body, action);
      list.append(card);
    }
    nodes.push(list, help, element("div", "note", "Apps with an Ollama setting work too. Use the address above as the server."));
    page.replaceChildren(...nodes);
  }
  function launchTool(app, model, target) {
    if (!native({ action: "terminal", tool: app.id, model })) copy(`ingot launch ${app.id} --model ${model}`, target);
  }

  window.addEventListener("ingot-view", (e) => {
    if (e.detail === "settings") openSettings();
    if (e.detail === "connect") openConnect();
  });
})();
