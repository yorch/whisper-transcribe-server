"use strict";
/* el, esc, tokenStore and makeApi come from common.js. The audit token is
   stored under the same key as /audit, so unlocking one page unlocks the other
   -- there is only one audit credential. */

const TOKENS = tokenStore("atk");

/* Which of the three modes the server rendered. Inferring it from a failed
   request is what made a switched-off API look like a bad token. */
const MODE = document.body.dataset.mode;

let unlocked = false, DATES = [];

function relock(){
  unlocked = false;
  el("main").classList.add("locked");
  // Only "off" owns the switched-off notice; anything else here is a credential
  // problem, where the gate is the useful thing to show.
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

function fmtHours(seconds){
  const h = (Number(seconds) || 0) / 3600;
  return (h >= 10 ? h.toFixed(1) : h.toFixed(2)) + " h";
}

function fmtBytes(n){
  n = Number(n) || 0;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while(n >= 1024 && i < units.length - 1){ n /= 1024; i += 1; }
  return (i ? n.toFixed(1) : String(n)) + " " + units[i];
}

/* Every value below is escaped by esc() before it reaches the markup; the
   totals are the only thing this page renders, and none of them is text. */
function cards(node, entries){
  const html = entries
    .filter((row) => row && row[1] !== null && row[1] !== undefined && row[1] !== "")
    .map(([label, value, hint]) =>
      '<div class="card"><b>' + esc(label) + "</b><span>" + esc(value) + "</span>"
      + (hint ? "<i>" + esc(hint) + "</i>" : "") + "</div>")
    .join("");
  // pi-lens-ignore: no-inner-html
  node.innerHTML = html;
}

function totals(agg, firstPassHint){
  return [
    ["Recordings", agg.recordings, firstPassHint],
    ["Distinct audio", fmtHours(agg.recording_seconds), "recordings, once each"],
    ["Audio processed", fmtHours(agg.audio_seconds), "includes retries"],
    ["Processing time", fmtHours(agg.processing_seconds), "wall time, includes model load"],
    ["Exports", agg.exports],
    ["Failed", agg.failed],
    ["Uploaded", fmtBytes(agg.uploaded_bytes)],
  ];
}

function pairList(prefix, mapping){
  const entries = Object.entries(mapping || {});
  if(!entries.length) return "";
  return entries
    .map(([k, v]) => (prefix ? k.replace(prefix, "") : k) + " " + v)
    .join(", ");
}

function operatorCards(agg){
  return [
    ["Refused jobs", agg.rejected, "queue full or too large"],
    ["Relabels", agg.relabels, "speaker pass, no transcription"],
    ["Diarize failures", agg.diarize_failed, "finished unlabelled"],
    ["Cancelled", agg.cancelled],
    ["Cap hits", agg.cap_hits, "a day file reached the byte cap"],
    ["Sidecars pruned", agg.sidecars_pruned],
    ["Collapsed", agg.suppressed, "refusals/repeats summarised"],
    ["Dropped by the trail", agg.lost, "events it could not write"],
    ["Refusals by kind", pairList("security.", agg.security), "authentication, host, cross-site"],
    ["Models", pairList("", agg.models), "completed passes"],
    ["Languages", pairList("", agg.languages)],
  ];
}

function integrityCards(data){
  const agg = data.retained || {};
  return [
    ["Recording", data.recording ? "on" : "off",
      data.recording ? "history covers the retained window" : "--no-audit: no history"],
    ["Retention", data.retain_days ? data.retain_days + " days" : "unlimited"],
    ["Day file cap", data.max_mb ? data.max_mb + " MB" : "none"],
    ["Sidecar cap", data.max_sidecars || "unlimited"],
    ["Trail dropped", data.lost_total, "events never written"],
    ["Sidecar dropped", data.sidecar_lost_total, "prompts not stored"],
    ["Audit degraded", data.degraded ? "yes" : "no"],
    ["Accounts for", (agg.events || 0) + " record(s)"],
  ];
}

function dayRow(d){
  return "<tr><td>" + esc(d.date) + "</td><td>" + esc(d.recordings) + "</td><td>"
    + esc(fmtHours(d.recording_seconds)) + "</td><td>" + esc(fmtHours(d.audio_seconds))
    + "</td><td>" + esc(d.relabels) + "</td><td>" + esc(d.failed) + "</td><td>"
    + esc(d.exports) + "</td><td>" + esc(d.rejected) + "</td></tr>";
}

async function load(){
  if(!unlocked) return;
  el("verify-out").textContent = "";
  try{
    const r = await api("/api/stats");
    if(!r.ok){
      let detail = r.statusText;
      try{ detail = (await r.json()).detail || detail; }catch{}
      throw new Error(detail);
    }
    const data = await r.json();
    unlocked = true;
    el("gate").classList.add("locked");
    el("main").classList.remove("locked");

    const rows = data.days || [];
    DATES = rows.map((d) => d.date);

    cards(el("session"), totals(data.session, "first passes this run"));
    // The operator and integrity views describe the retained trail, not this
    // run, so they come from the summed days.
    cards(el("operator"), operatorCards(data.retained || {}));
    cards(el("integrity"), integrityCards(data));

    // Every field is escaped inside dayRow().
    // pi-lens-ignore: no-inner-html
    el("day-rows").innerHTML = rows.map(dayRow).join("");
    el("days").classList.toggle("locked", rows.length === 0);
    el("empty").classList.toggle("locked", rows.length > 0);

    const since = new Date((data.session_since || 0) * 1000);
    el("meta").textContent = "This run since " + since.toISOString().replace("T", " ").slice(0, 19)
      + " UTC \u00b7 " + rows.length + " retained day(s)"
      + " \u00b7 recorded reads make today's row move";
  }catch(err){
    if(err.message === "unauthorised") return;
    el("meta").textContent = "Could not load the statistics: " + err.message;
  }
}

async function verifyAll(){
  const out = el("verify-out");
  if(!DATES.length){ out.textContent = "No retained days to verify."; return; }
  out.textContent = "Verifying " + DATES.length + " day(s)\u2026";
  const bad = [];
  for(const day of DATES){
    try{
      const r = await api("/api/audit/verify?date=" + encodeURIComponent(day));
      if(!r.ok) throw new Error(r.statusText);
      const v = await r.json();
      if(!v.ok) bad.push(day + " at line " + v.first_bad_line + ": " + v.reason);
    }catch(err){
      if(err.message === "unauthorised") return;
      bad.push(day + ": " + err.message);
    }
  }
  out.textContent = bad.length
    ? "Chain problems \u2014 " + bad.join("; ")
    : "All " + DATES.length + " retained day(s) verify.";
}

el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKENS.set(value);
  try{
    const r = await fetch("/api/stats", {headers:{"x-audit-token":value}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    el("gate-err").classList.add("locked");
    el("gate-token").value = "";
    unlocked = true;
    await load();
  }catch{
    el("gate-err").classList.remove("locked");
  }
}

el("reload").addEventListener("click", load);
el("verify").addEventListener("click", verifyAll);

(async function boot(){
  // No early return on a missing token: --audit-open has none to have, and this
  // one probe is what tells the three modes apart.
  try{
    const headers = TOKENS.get() ? {"x-audit-token":TOKENS.get()} : {};
    const r = await fetch("/api/stats", {headers, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    unlocked = true;
    await load();
  }catch{
    relock();
  }
})();
