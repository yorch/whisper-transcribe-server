"use strict";
/* el, esc, tokenStore and makeApi come from common.js, which both pages share.
   The audit page keeps its own storage key and header so its credential cannot
   be confused with the app token. */

const TOKENS = tokenStore("atk");
let unlocked = false, offset = 0, timer = null;

const api = makeApi({
  header: "x-audit-token",
  store: TOKENS,
  /* 404 too: the audit API answers that when no audit token is configured. */
  isUnauthorised: (status) => status === 401 || status === 404,
  onUnauthorised: () => {
    unlocked = false;
    el("gate").classList.remove("locked");
    el("main").classList.add("locked");
  },
});
function level(event){
  if(!event) return "";
  if(event.startsWith("security.") || event === "request.rejected" || event.endsWith("error")) return "bad";
  if(event.endsWith("rejected")) return "warn";
  if(event === "job.done" || event === "server.started") return "good";
  return "";
}

const SKIP = new Set(["ts","event","client","client_claimed","host","method","path",
                      "prompt_text","hotwords_text"]);

function fmtVal(v){
  if(v === null || v === undefined) return "";
  if(typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function row(rec){
  const time = String(rec.ts || "").slice(11, 19);
  const where = [];
  if(rec.method) where.push(rec.method + " " + (rec.path || ""));
  // client_claimed is whatever the proxy said; the TCP peer is the fact.
  if(rec.client_claimed) where.push("unverified via " + rec.client_claimed);
  if(rec.client) where.push(rec.client);
  if(rec.host) where.push(rec.host);
  if(rec.status) where.push("HTTP " + rec.status);
  if(rec.ms !== undefined && rec.ms !== null) where.push(rec.ms + "ms");

  const chips = Object.entries(rec)
    .filter(([k, v]) => !SKIP.has(k) && v !== null && v !== undefined && v !== "")
    .map(([k, v]) => '<span class="chip"><b>' + esc(k) + "</b>" + esc(fmtVal(v)) + "</span>")
    .join("");

  let prompt = "";
  if(rec.hotwords_text)
    prompt += '<div><b>hotwords</b>' + esc(rec.hotwords_text) + "</div>";
  if(rec.prompt_text)
    prompt += '<div><b>prompt</b>' + esc(rec.prompt_text) + "</div>";

  return '<div class="ev ' + level(rec.event) + '">' +
    '<div class="ev-head">' +
      '<span class="ev-time">' + esc(time) + "</span>" +
      '<span class="ev-event">' + esc(rec.event) + "</span>" +
      '<span class="ev-src">' + esc(where.join(" \u00b7 ")) + "</span>" +
    "</div>" +
    (chips ? '<div class="chips">' + chips + "</div>" : "") +
    (prompt ? '<div class="prompt">' + prompt + "</div>" : "") +
  "</div>";
}

function fillDates(dates, selected){
  const node = el("date");
  const current = node.value;
  const wanted = selected || current;
  node.textContent = "";
  if(!dates.length){
    const o = document.createElement("option");
    o.value = o.textContent = selected || "";
    node.append(o);
    return;
  }
  for(const d of dates){
    const o = document.createElement("option");
    o.value = o.textContent = d;
    if(d === wanted) o.selected = true;
    node.append(o);
  }
}

async function load(more){
  if(!unlocked) return;
  if(!more) offset = 0;
  const params = new URLSearchParams();
  params.set("date", el("date").value);
  params.set("limit", el("limit").value);
  params.set("offset", String(offset));
  if(el("job").value.trim()) params.set("job", el("job").value.trim());
  if(el("q").value.trim()) params.set("q", el("q").value.trim());
  if(el("prompts").checked) params.set("include_prompts", "1");

  try{
    const r = await api("/api/audit?" + params.toString());
    if(!r.ok){
      let detail = r.statusText;
      try{ detail = (await r.json()).detail || detail; }catch(_){}
      throw new Error(detail);
    }
    const data = await r.json();
    unlocked = true;
    el("gate").classList.add("locked");
    el("main").classList.remove("locked");
    fillDates(data.dates, data.date);

    if(!more) el("events").textContent = "";
    el("events").insertAdjacentHTML("beforeend", data.events.map(row).join(""));
    offset += data.events.length;

    el("empty").classList.toggle("locked", data.total > 0);
    el("more").disabled = offset >= data.total;
    el("meta").textContent = data.total + " event(s) on " + data.date
      + " \u00b7 showing " + Math.min(offset, data.total)
      + " \u00b7 retention " + (data.retain_days ? data.retain_days + " days" : "unlimited")
      + (data.prompts_available ? "" : " \u00b7 prompts not stored");
  }catch(err){
    if(err.message === "unauthorised") return;
    el("meta").textContent = "Could not load the audit trail: " + err.message;
  }
}

function schedule(){
  clearInterval(timer);
  if(el("auto").checked) timer = setInterval(() => {
    if(offset <= Number(el("limit").value)) load(false);
  }, 5000);
}

el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKENS.set(value);
  try{
    const r = await fetch("/api/audit?limit=1", {headers:{"x-audit-token":value}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    el("gate-err").classList.add("locked");
    el("gate-token").value = "";
    unlocked = true;
    await load(false);
    schedule();
  }catch(e){
    el("gate-err").classList.remove("locked");
  }
}

el("reload").addEventListener("click", () => load(false));
// Changing the day starts a fresh page; "Load more" only ever appends.
el("more").addEventListener("click", () => load(true));
el("date").addEventListener("change", () => load(false));
el("job").addEventListener("change", () => load(false));
el("prompts").addEventListener("change", () => load(false));
el("q").addEventListener("keydown", (e) => { if(e.key === "Enter") load(false); });
el("auto").addEventListener("change", schedule);

(async function boot(){
  if(!TOKENS.get()) return;
  try{
    const r = await fetch("/api/audit?limit=1", {headers:{"x-audit-token":value}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    unlocked = true;
    await load(false);
    schedule();
  }catch(e){
    el("gate").classList.remove("locked");
  }
})();
