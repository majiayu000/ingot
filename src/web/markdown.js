// Markdown for answers. Everything is HTML-escaped before tags are added.
"use strict";
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
// Code highlighting: comments, strings, numbers and keywords, by language
// family. Tokens are found in the raw code and escaped one by one.
const HASH_COMMENTS = new Set(["python", "py", "sh", "bash", "zsh", "shell", "ruby", "rb", "yaml", "yml", "toml", "r", "perl", "dockerfile", "makefile"]);
const KEYWORDS = new Set(("and as assert async await break case catch class const continue def default defer del do elif else enum " +
  "except export extends false final finally fn for from func function go if impl import in interface is lambda let loop match mod " +
  "module mut new nil none not null or package pass private protected pub public raise return self static struct super switch " +
  "this throw trait true try type typeof undefined use var void where while with yield None True False echo then fi done esac").split(" "));
const STRING = String.raw`"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|` + "`(?:\\\\.|[^`\\\\])*`";
function highlight(code, lang) {
  lang = lang.toLowerCase();
  if (!lang || lang === "code" || lang === "text" || lang === "plaintext" || code.length > 60000) return escapeHtml(code);
  const comment = HASH_COMMENTS.has(lang) ? String.raw`#[^\n]*` : lang === "sql" || lang === "lua" ? String.raw`--[^\n]*` : String.raw`\/\/[^\n]*|\/\*[\s\S]*?\*\/`;
  const pattern = new RegExp(`(${comment})|(${STRING})|` + String.raw`\b(\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b`, "g");
  let out = "", last = 0;
  for (const m of code.matchAll(pattern)) {
    const kind = m[1] ? "c" : m[2] ? "s" : m[3] ? "n" : KEYWORDS.has(m[4]) ? "k" : "";
    if (!kind) continue;
    out += escapeHtml(code.slice(last, m.index)) + `<span class="tok-${kind}">` + escapeHtml(m[0]) + "</span>";
    last = m.index + m[0].length;
  }
  return out + escapeHtml(code.slice(last));
}
const unescapeHtml = (text) => text.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, "&");
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
        '</span><button class="icon copy-code">Copy</button></div><pre><code>' + highlight(unescapeHtml(code.join("\n")), fence[1]) + "</code></pre></div>");
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
