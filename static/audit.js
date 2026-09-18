"use strict";
/* el, esc, tokenStore and makeApi come from common.js, which both pages share.
   The audit page keeps its own storage key and header so its credential cannot
   be confused with the app token. */

const TOKENS = tokenStore("atk");

/* Which of the three modes the server rendered. The page used to infer this
   from a failed request, which made a switched-off API look like a bad token. */
const MODE = document.body.dataset.mode;

let unlocked = false, offset = 0, cursor = null, timer = null;

function relock(){
  unlocked = false;
  clearInterval(timer);
  el("main").classList.add("locked");
  // Only "off" owns the switched-off notice; anything else that lands here is
  // a credential problem, where the gate is the useful thing to show.
  el("gate").classList.toggle("locked", MODE === "off");
  el("off").classList.toggle("locked", MODE !== "off");
}

const api = makeApi({
  header: "x-audit-token",
  store: TOKENS,
  /* 404 too: the audit API answers that when it is switched off. */
  isUnauthorised: (status) => status === 401 || status === 404,
  onUnauthorised: relock,
});
function level(event){
  if(!event) return "";
  if(event.startsWith("security.") || event === "request.rejected" || event.endsWith("error")) return "bad";
  if(event.endsWith("rejected")) return "warn";
  if(event === "job.done" || event === "server.started") return "good";
  return "";
}

const SKIP = new Set(["ts","event","client","client_claimed","host","method","path",
                      "prompt_text","hotwords_text","seq","prev","chain","carry"]);

function fmtVal(v){
  if(v === null || v === undefined) return "";
  if(typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function row(rec){
  // Times are UTC, like the day files. Say so: an operator reads an unlabelled
  // clock as local time and mis-orders a day.
  const stamp = String(rec.ts || "");
  const time = (stamp ? stamp.slice(11, 19) + " UTC" : "")
    + (rec.seq ? " \u00b7 #" + rec.seq : "");
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
  if(!more){ offset = 0; cursor = null; }
  const params = new URLSearchParams();
  params.set("date", el("date").value);
  params.set("limit", el("limit").value);
  /* The cursor is a line number from the front of the file, so it does not move
     when new records land; an offset into a newest-first window does, which
     duplicates or skips a row. Falls back to the offset on the first page. */
  if(more && cursor) params.set("before_line", String(cursor));
  else params.set("offset", String(offset));
  if(el("job").value.trim()) params.set("job", el("job").value.trim());
  if(el("q").value.trim()) params.set("q", el("q").value.trim());
  if(el("prompts").checked) params.set("include_prompts", "1");

  try{
    const r = await api("/api/audit?" + params.toString());
    if(!r.ok){
      let detail = r.statusText;
      try{ detail = (await r.json()).detail || detail; }catch{}
      throw new Error(detail);
    }
    const data = await r.json();
    unlocked = true;
    el("gate").classList.add("locked");
    el("main").classList.remove("locked");
    fillDates(data.dates, data.date);

    if(!more) el("events").textContent = "";
    // The page's one deliberate HTML sink. Every field row() interpolates --
    // event name, source, chips, prompt text -- goes through esc(), which is
    // the boundary the CSP and the whole static/ split exist to keep honest.
    // pi-lens-ignore: no-inner-html
    el("events").insertAdjacentHTML("beforeend", data.events.map(row).join(""));
    offset += data.events.length;
    cursor = data.next_before_line;

    el("empty").classList.toggle("locked", data.total > 0);
    el("more").disabled = !data.has_more;
    el("meta").textContent = data.total + " event(s) on " + data.date
      + " UTC \u00b7 showing " + Math.min(offset, data.total)
      + " \u00b7 retention " + (data.retain_days ? data.retain_days + " days" : "unlimited")
      + (data.prompts_available ? "" : " \u00b7 prompts not stored");
    // A sink that is failing still serves reads, so the page can -- and must --
    // say the trail it is showing has holes in it.
    const warn = el("degraded");
    warn.classList.toggle("locked", !data.degraded);
    if(data.degraded){
      const bits = [];
      if(data.lost_total) bits.push(data.lost_total + " event(s) dropped");
      if(data.sidecar_lost_total) bits.push(data.sidecar_lost_total + " prompt sidecar(s) not written");
      if(data.last_error) bits.push(data.last_error);
      warn.textContent = "The audit trail is not keeping up: " + (bits.join(" \u00b7 ") || "unknown")
        + ". Events are being lost until the sink is fixed.";
    }
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

async function verify(){
  const out = el("verify-out");
  out.textContent = "Verifying\u2026";
  try{
    const day = el("date").value;
    const r = await api("/api/audit/verify?date=" + encodeURIComponent(day));
    if(!r.ok){
      let detail = r.statusText;
      try{ detail = (await r.json()).detail || detail; }catch{}
      throw new Error(detail);
    }
    const v = await r.json();
    const head = v.head ? v.head.slice(0, 12) + "\u2026" : "none";
    if(!v.ok){
      out.textContent = "Chain broken on " + v.date + " at line " + v.first_bad_line
        + " (expected #" + v.first_bad_seq + "): " + v.reason;
      return;
    }
    if(!v.checked){
      out.textContent = "Nothing to verify on " + v.date + ": "
        + v.legacy + " record(s) predate the chain.";
      return;
    }
    let text = "Verified " + v.checked + " record(s) on " + v.date
      + (v.legacy ? " (" + v.legacy + " legacy skipped)" : "")
      + " \u00b7 head " + head;
    if(v.carry_ok === true) text += " \u00b7 linked to the previous day";
    else if(v.carry_ok === false) text += " \u00b7 carry does NOT match the previous day";
    out.textContent = text;
  }catch(err){
    if(err.message === "unauthorised") return;
    out.textContent = "Could not verify: " + err.message;
  }
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
  }catch{
    el("gate-err").classList.remove("locked");
  }
}

el("reload").addEventListener("click", () => load(false));
el("verify").addEventListener("click", verify);
// Changing the day starts a fresh page; "Load more" only ever appends.
el("more").addEventListener("click", () => load(true));
el("date").addEventListener("change", () => load(false));
el("job").addEventListener("change", () => load(false));
el("prompts").addEventListener("change", () => load(false));
el("q").addEventListener("keydown", (e) => { if(e.key === "Enter") load(false); });
el("auto").addEventListener("change", schedule);

(async function boot(){
  // No early return on a missing token: --audit-open means there is none to
  // have, and this one probe is what tells the three modes apart.
  try{
    const headers = TOKENS.get() ? {"x-audit-token":TOKENS.get()} : {};
    const r = await fetch("/api/audit?limit=1", {headers, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    unlocked = true;
    await load(false);
    schedule();
  }catch{
    relock();
  }
})();
