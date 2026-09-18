"use strict";
/* el, esc, tokenStore and makeApi come from common.js, which both pages share. */

const TOKENS = tokenStore("tk");
let unlocked = false;

const api = makeApi({
  header: "x-token",
  store: TOKENS,
  isUnauthorised: (status) => status === 401,
  onUnauthorised: () => {
    unlocked = false;
    el("gate").classList.remove("locked");
    el("main").classList.add("locked");
  },
});
el("gate-go").addEventListener("click", submitToken);
el("gate-token").addEventListener("keydown", (e) => { if(e.key === "Enter") submitToken(); });

async function submitToken(){
  const value = el("gate-token").value.trim();
  if(!value) return;
  TOKENS.set(value);
  try{
    const r = await fetch("/api/status", {headers:{"x-token":value}, credentials:"omit"});
    if(!r.ok) throw new Error("rejected");
    el("gate-err").classList.add("locked");
    el("gate").classList.add("locked");
    el("gate-token").value = "";
    await boot();
  }catch(e){
    el("gate-err").classList.remove("locked");
  }
}

/* ---------- status strip ---------- */
/* Which card, and how much of it is in use. Everything else the server knows
   lives in the tooltip: a five-cell strip that tries to show eight things shows
   nothing. Both functions are pure so the composition can be tested without a
   browser. */
function gb(mb){
  return (mb / 1024).toFixed(1);
}

function fmtGb(mb){
  return gb(mb) + " GB";
}

function deviceLabel(s){
  const d = s.device_info || {};
  const base = d.name || s.gpu || s.device;
  // "1.7/12.0 GB" rather than "1.7 GB/12.0 GB": the unit belongs to the pair.
  if(d.vram_total_mb && d.vram_used_mb != null)
    return base + " \u00b7 " + gb(d.vram_used_mb) + "/" + fmtGb(d.vram_total_mb);
  return base;
}

function deviceTooltip(s){
  const d = s.device_info || {};
  const cuda = s.cuda || null;
  const lines = [];
  // Why it is not using the GPU is the first thing anyone hovers for.
  if(cuda && !cuda.usable) lines.push(cuda.reason || "CUDA unavailable");
  if(d.name) lines.push(d.name);
  if(d.vram_total_mb)
    lines.push("VRAM " + (d.vram_used_mb != null ? fmtGb(d.vram_used_mb) + " used of " : "")
               + fmtGb(d.vram_total_mb));
  // Distinct from the total: this is what the server itself is holding, which
  // is what the model cache and a running job account for.
  if(d.process_mb != null) lines.push("this server holds " + fmtGb(d.process_mb));
  if(d.driver) lines.push("driver " + d.driver);
  if(d.compute_cap) lines.push("compute capability " + d.compute_cap
    + (parseFloat(d.compute_cap) < 7 ? " \u2014 no fast fp16, use int8" : ""));
  if(d.count) lines.push(d.count + " CUDA device(s)");
  lines.push("running on " + s.device + " at " + s.compute_type);
  if(d.python) lines.push("Python " + d.python + " on " + d.platform);
  if(!d.name && cuda && cuda.usable)
    lines.push("nvidia-smi not found, so the card is unnamed");
  return lines.join("\n");
}

async function refreshStatus(){
  try{
    const s = await (await api("/api/status")).json();
    unlocked = true;
    el("main").classList.remove("locked");
    el("gate").classList.add("locked");

    el("lamp").className = "lamp " + (s.device === "cuda" ? "on" : "bad");
    el("r-device").textContent = deviceLabel(s);
    el("r-device").title = deviceTooltip(s);
    el("r-precision").textContent = s.compute_type;
    el("r-ffmpeg").textContent = s.ffmpeg ? "found" : "missing";
    el("r-ffmpeg").classList.toggle("bad", !s.ffmpeg);
    el("r-queue").textContent = s.active_jobs;
    el("r-loaded").textContent = (s.loaded_models && s.loaded_models.length)
      ? s.loaded_models.join(", ") : "none";
    MAX_MB = s.max_upload_mb;
    RETRY_OK = s.retry_available;
    DIARIZE_OK = !!s.allow_diarize;

    if(!POPULATED){
      POPULATED = true;
      fill("model", s.models, s.default_model);
      fill("quality", s.qualities, s.default_quality);
      fill("compute", s.compute_types, s.compute_type);
      // Precision is a property of this machine, not of a recording. It only
      // appears if the operator explicitly opened it up.
      if(s.allow_precision_choice) el("compute-field").classList.remove("locked");
      if(!DIARIZE_OK){
        el("diarize").checked = false;
        el("diarize-row").classList.add("locked");
        el("diarize-hint").classList.add("locked");
      }
      /* Now that DIARIZE_OK is known, apply the word-timing coupling -- or
         leave that box alone if the server has no diarization to couple it
         to. */
      syncDiarize();
      if(!s.allow_model_choice){
        el("model").disabled = true;
        el("model").title = "Pinned by the server";
      }
    }
  }catch(e){
    if(e.message !== "unauthorised"){
      el("lamp").className = "lamp bad";
      el("r-device").textContent = "server unreachable";
      el("r-device").title = "";
    }
  }
}

/* ---------- upload ---------- */
let MAX_MB = 0, RETRY_OK = true, POPULATED = false, DIARIZE_OK = false;

function fill(id, values, selected){
  const node = el(id);
  node.textContent = "";
  for(const v of values){
    const o = document.createElement("option");
    o.value = o.textContent = v;
    if(v === selected) o.selected = true;
    node.append(o);
  }
}

/* Whatever is in the controls right now, as a form body. Used for both a fresh
   upload and a retry, so a retry re-runs the same audio under new settings. */
function currentSettings(){
  const fd = new FormData();
  fd.append("model", el("model").value);
  fd.append("compute_type", el("compute").value);
  fd.append("language", el("language").value);
  fd.append("quality", el("quality").value);
  fd.append("vad", el("vad").checked ? "true" : "false");
  fd.append("prompt", el("prompt").value);
  fd.append("hotwords", el("hotwords").value);
  fd.append("translate", el("translate").checked ? "true" : "false");
  fd.append("condition", el("condition").checked ? "true" : "false");
  fd.append("word_timestamps", el("words").checked ? "true" : "false");
  fd.append("diarize", (DIARIZE_OK && el("diarize").checked) ? "true" : "false");
  fd.append("speakers", el("speakers").value || "0");
  fd.append("min_silence_ms", el("min-silence").value || "2000");
  fd.append("speech_pad_ms", el("speech-pad").value || "400");
  return fd;
}

const intake = el("intake"), picker = el("picker");
intake.addEventListener("click", () => picker.click());
intake.addEventListener("keydown", (e) => {
  if(e.key === "Enter" || e.key === " "){ e.preventDefault(); picker.click(); }
});
picker.addEventListener("change", () => { send([...picker.files]); picker.value = ""; });

for(const ev of ["dragenter","dragover"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.add("hot"); });
for(const ev of ["dragleave","drop"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.remove("hot"); });
intake.addEventListener("drop", (e) => send([...e.dataTransfer.files]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

async function send(files){
  for(const f of files){
    if(MAX_MB && f.size > MAX_MB * 1024 * 1024){
      alert(f.name + " is larger than the " + MAX_MB + " MB limit.");
      continue;
    }
    const fd = currentSettings();
    fd.append("file", f);
    try{
      const r = await api("/api/jobs", {method:"POST", body:fd});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        throw new Error(detail);
      }
      await r.json();
      await tick();
    }catch(err){
      if(err.message !== "unauthorised") alert("Upload failed for " + f.name + ": " + err.message);
    }
  }
}

el("vad").addEventListener("change", () => {
  el("vad-tuning").classList.toggle("locked", !el("vad").checked);
});

/* Asking for speaker labels asks for word timings, because without them a whole
   Whisper segment can only be handed to its dominant speaker — which is the
   version of this feature that is confidently wrong. The box is ticked and
   locked rather than quietly overridden, so the cost is visible.

   Split out from the DOM work because the interesting part is the decision: a
   server started with --no-diarize hides the control, and the page must not
   then leave word timings ticked and disabled behind it. A separate function
   is one that can be tested without a browser. */
function locksWordTimings(diarizeAvailable, boxTicked){
  return !!diarizeAvailable && !!boxTicked;
}

function syncDiarize(){
  const on = locksWordTimings(DIARIZE_OK, el("diarize").checked);
  if(on) el("words").checked = true;
  el("words").disabled = on;
  el("words").title = on ? "Required for speaker labels" : "";
}
el("diarize").addEventListener("change", syncDiarize);
/* Deliberately not called here: DIARIZE_OK is only known once /api/status has
   answered, and running this before then would lock the word-timing box for a
   feature the server may not offer. Called from the status handler instead. */

/* ---------- following the transcript ---------- */
/* Auto-scroll for the job previews, on by default and remembered for the
   session: a switch that quietly comes back on after every reload is worse
   than not having one. */
const FOLLOW_KEY = "follow";
try{ el("follow").checked = sessionStorage.getItem(FOLLOW_KEY) !== "0"; }catch(_){}
el("follow").addEventListener("change", () => setFollow(el("follow").checked));

/* ---------- rendering ---------- */
const TICKS = 40;
const known = new Map();
/* One entry per card. The transcript keeps its DOM between polls so a running
   job can be read while it grows: replacing the whole card every 1.2s (what
   this used to do) threw away the scroll position and any selection. */
const views = new Map();

function meter(job){
  const lit = Math.round((job.progress || 0) * TICKS);
  const cls = job.state === "done" ? "done" : job.state === "error" ? "fail" : "lit";
  let html = '<div class="meter">';
  for(let i=0;i<TICKS;i++) html += '<i class="' + (i<lit?cls:"") + '"></i>';
  return html + "</div>";
}

function fmtTime(s){
  s = Math.round(s);
  const m = Math.floor(s/60);
  return m ? m + "m " + String(s%60).padStart(2,"0") + "s" : s + "s";
}

function statusLine(job){
  if(job.state === "queued")  return "Waiting for the GPU";
  if(job.state === "loading") return "Loading " + esc(job.opts.model);
  if(job.state === "running"){
    // The diarization pass has no useful ETA of its own, and the meter is in
    // its last tenth by then, so report the phase rather than a wrong guess.
    if(job.phase === "diarizing") return esc(job.message || "Identifying speakers");
    const pct = Math.round((job.progress||0)*100);
    const eta = job.progress > 0.02
      ? " \u00b7 about " + fmtTime(job.elapsed/job.progress - job.elapsed) + " left"
      : "";
    return pct + "% \u00b7 " + job.segment_count + " segments" + eta;
  }
  if(job.state === "done"){
    const speed = job.duration && job.elapsed
      ? " \u00b7 " + (job.duration/job.elapsed).toFixed(1) + "\u00d7 realtime" : "";
    return "Finished in " + fmtTime(job.elapsed) + speed
      + " \u00b7 " + esc(job.language || "?") + " \u00b7 " + job.segment_count + " segments";
  }
  if(job.state === "cancelled") return "Cancelled";
  return esc(job.message);
}

/* Only show the knobs that were actually off-default, so the line stays short. */
function jobTags(job){
  const o = job.opts || {};
  const bits = [o.model];
  if(o.quality && o.quality !== "balanced") bits.push(o.quality);
  if(o.translate) bits.push("translated");
  if(o.condition) bits.push("context on");
  if(o.word_timestamps) bits.push("word times");
  if(o.has_hotwords) bits.push("terms");
  const found = new Set((job.segments || [])
    .map(s => s.speaker).filter(v => v != null));
  if(o.diarize && found.size)
    bits.push(found.size + (found.size === 1 ? " speaker" : " speakers"));
  if(job.duration) bits.push(fmtTime(job.duration));
  return bits.join(" \u00b7 ");
}

/* Seconds to HH:MM:SS: the clock of the timestamped export without the
   milliseconds, which are noise on screen. */
function fmtStamp(seconds){
  const total = Math.max(0, Math.floor(seconds || 0));
  const pad = (n) => String(n).padStart(2,"0");
  return pad(Math.floor(total/3600)) + ":" + pad(Math.floor(total/60)%60)
       + ":" + pad(total%60);
}

function followOn(){ return el("follow").checked; }

function setPaused(view, paused){
  view.paused = paused;
  view.transcript.classList.toggle("paused", paused);
  view.transcript.title = paused
    ? "Following paused \u2014 scroll back to the bottom to catch up" : "";
}

/* Keep the newest line in view, unless Following is off or the operator has
   scrolled away from the bottom. */
function stick(view){
  if(!followOn() || view.paused) return;
  view.transcript.scrollTop = view.transcript.scrollHeight;
}

function setFollow(on){
  try{ sessionStorage.setItem(FOLLOW_KEY, on ? "1" : "0"); }catch(_){}
  for(const view of views.values()){
    setPaused(view, false);
    if(on && view.live) stick(view);   // catch up now, not at the next segment
  }
}

function createCard(job){
  const node = document.createElement("div");
  node.className = "job";
  node.id = "job-" + job.id;
  node.innerHTML =
    '<div class="job-head"></div>' +
    '<div class="meter-slot"></div>' +
    '<div class="job-body">' +
      '<p class="status"></p>' +
      '<div class="transcript"></div>' +
      '<div class="actions"></div>' +
    "</div>";
  el("jobs").prepend(node);

  const view = {
    node,
    transcript: node.querySelector(".transcript"),
    shown: 0,
    paused: false,
    live: false,
    labeled: false,
  };
  /* Scrolling away from the newest line pauses; coming back re-arms. The
     Follow switch stays the authoritative off switch. */
  view.transcript.addEventListener("scroll", () => {
    const atBottom = view.transcript.scrollHeight - view.transcript.scrollTop
                     - view.transcript.clientHeight <= 8;
    if(atBottom === view.paused) setPaused(view, !atBottom);
  });
  views.set(job.id, view);
  return view;
}

/* Append only what is new. The server grows job["segments"] and never rewrites
   one, so counting is enough. textContent rather than innerHTML: transcript
   text needs no escaping and cannot become markup. */
function appendSegments(view, segments, live, total, labeled){
  view.live = live;
  let rebuilt = false;
  /* `segments` is the tail the server has not sent yet, so there is nothing to
     de-duplicate: a poll with no new text sends an empty list. `total` is the
     server's own count, which is the only way to notice that a job's segment
     list went backwards (a restart, or a re-queued job) and the preview has to
     be rebuilt rather than appended to. */
  if(total !== undefined && total < view.shown + segments.length){
    view.transcript.textContent = "";
    view.shown = 0;
    rebuilt = true;
  }
  /* Labels arrive in one batch when the diarization pass ends, so every row
     already on screen was drawn without them, and counting new segments cannot
     notice. The server reports the shape, because a tail carrying no new
     segments would otherwise never reveal it, and the caller refetches from
     zero when this fires — a tail cannot rebuild rows already drawn. */
  if(labeled !== undefined && labeled !== view.labeled){
    view.transcript.textContent = "";
    view.shown = 0;
    view.labeled = labeled;
    rebuilt = true;
  }
  for(const seg of segments){
    const row = document.createElement("div");
    row.className = "seg";
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = fmtStamp(seg.start);
    const sp = document.createElement("span");
    sp.className = "sp";
    if(seg.speaker != null){
      sp.textContent = "Speaker " + seg.speaker;
      // Three colours, cycled: enough to tell voices apart at a glance while
      // staying readable, and no legend to maintain.
      sp.classList.add("s" + ((seg.speaker - 1) % 3));
      sp.title = sp.textContent;
    }
    const tx = document.createElement("span");
    tx.className = "tx";
    tx.textContent = seg.text;
    row.append(ts, sp, tx);
    view.transcript.append(row);
  }
  view.shown += segments.length;
  /* Only chase a job that is still producing text: a finished transcript should
     open at its first line, not at its last. A rebuild is the exception —
     clearing the transcript resets the scroll, so someone who was following a
     live job would be thrown back to the top the moment the labels landed. */
  if(live || rebuilt) stick(view);
}

function actionsHtml(job){
  const dl = (fmt, label) =>
    '<button data-dl="' + fmt + '" data-id="' + esc(job.id) + '">' + label + "</button>";
  return (job.state === "done"
      ? '<button data-copy="' + esc(job.id) + '">Copy text</button>' +
        dl("txt","Save .txt") + dl("timestamped","Save timestamped") +
        dl("srt","Save .srt") + dl("vtt","Save .vtt") + dl("json","Save .json")
      : "") +
    (RETRY_OK && job.can_retry
      ? '<button data-retry="' + esc(job.id) + '">Retry with these settings</button>'
      : "") +
    '<button class="ghost" data-del="' + esc(job.id) + '">' +
      (["queued","loading","running"].includes(job.state) ? "Cancel" : "Remove") +
    "</button>";
}

function render(job){
  const id = "job-" + job.id;
  let view = views.get(id);
  if(!view || !view.node.isConnected) view = createCard(job);

  const name = esc(job.filename);
  view.node.querySelector(".job-head").innerHTML =
    '<span class="job-name" title="' + name + '">' + name + "</span>" +
    '<span class="job-meta">' + esc(jobTags(job)) + "</span>";
  view.node.querySelector(".meter-slot").innerHTML = meter(job);
  const status = view.node.querySelector(".status");
  status.className = "status" + (job.state === "error" ? " err" : "");
  status.innerHTML = statusLine(job);
  view.node.querySelector(".actions").innerHTML = actionsHtml(job);

  appendSegments(view, job.segments || [],
                 ["queued","loading","running"].includes(job.state),
                 job.segment_count,
                 job.speaker_labels);
}

/* Downloads go through fetch so the token stays in a header, never a URL. */
async function download(jobId, fmt){
  const r = await api("/api/jobs/" + encodeURIComponent(jobId)
                      + "/text?format=" + encodeURIComponent(fmt) + "&download=1");
  if(!r.ok) return;
  const blob = await r.blob();
  const cd = r.headers.get("content-disposition") || "";
  let name = "transcript." + (fmt === "timestamped" ? "txt" : fmt);
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  const plain = /filename="([^"]*)"/i.exec(cd);
  if(star) { try { name = decodeURIComponent(star[1]); } catch(_){} }
  else if(plain) name = plain[1];
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

el("jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if(!b) return;
  try{
    if(b.dataset.dl) await download(b.dataset.id, b.dataset.dl);
    if(b.dataset.copy){
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.copy) + "/text?format=txt");
      const body = await r.text();
      if(navigator.clipboard && window.isSecureContext){
        await navigator.clipboard.writeText(body);
      }else{
        // Plain HTTP is not a secure context, so the Clipboard API is unavailable.
        const ta = document.createElement("textarea");
        ta.value = body;
        ta.className = "offscreen";
        document.body.append(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy text"), 1400);
    }
    if(b.dataset.retry){
      b.disabled = true;
      const r = await api("/api/jobs/" + encodeURIComponent(b.dataset.retry) + "/retry",
                          {method:"POST", body:currentSettings()});
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch(_){}
        alert("Retry failed: " + detail);
        b.disabled = false;
      }
      await tick();
    }
    if(b.dataset.del){
      await api("/api/jobs/" + encodeURIComponent(b.dataset.del), {method:"DELETE"});
      await tick();
    }
  }catch(err){ /* 401 already handled by api() */ }
});

/* ---------- poll ---------- */
async function tick(){
  if(!unlocked) return;
  try{
    const {jobs} = await (await api("/api/jobs")).json();
    const live = jobs.map(j => j.id);
    el("empty").classList.toggle("locked", jobs.length > 0);

    for(const stale of [...known.keys()].filter(id => !live.includes(id))){
      const n = document.getElementById("job-" + stale);
      if(n) n.remove();
      known.delete(stale);
      views.delete(stale);
    }

    for(const summary of jobs){
      const sig = summary.state + ":" + summary.progress + ":" + summary.segment_count;
      if(known.get(summary.id) === sig) continue;
      known.set(summary.id, sig);

      /* Ask for only what this card has not seen. A card that lost its rows
         (the job's segments went backwards) starts from zero again. */
      const view = views.get("job-" + summary.id);
      let since = view ? view.shown : 0;
      if(view && summary.segment_count < view.shown){
        view.transcript.textContent = "";
        view.shown = 0;
        since = 0;
      }
      const url = "/api/jobs/" + encodeURIComponent(summary.id);
      let full = await (await api(url + "?since=" + since)).json();
      /* Labels arrive in one batch when diarization finishes, and the tail
         cannot carry them: refetch the whole transcript once when the shape
         changes, so rows drawn without labels are rebuilt with them. */
      if(view && full.speaker_labels !== view.labeled && view.shown > 0){
        view.transcript.textContent = "";
        view.shown = 0;
        full = await (await api(url + "?since=0")).json();
      }
      render(full);
    }
  }catch(e){ /* server blip or 401; next tick retries */ }
}

async function boot(){
  await refreshStatus();
  await tick();
}

boot();
setInterval(tick, 1200);
setInterval(refreshStatus, 5000);
