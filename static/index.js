"use strict";
/* el, esc, tokenStore and makeApi come from common.js, which both pages share. */

const TOKENS = tokenStore("tk");
let unlocked = false;

/* A rejected token puts the gate back. Named, because the upload (XHR, not
   api()) has to do the same thing. */
function lockPage(){
  unlocked = false;
  el("gate").classList.remove("locked");
  el("main").classList.add("locked");
}

const api = makeApi({
  header: "x-token",
  store: TOKENS,
  isUnauthorised: (status) => status === 401,
  onUnauthorised: lockPage,
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
  }catch{
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
picker.addEventListener("change", () => { stage([...picker.files]); picker.value = ""; });

for(const ev of ["dragenter","dragover"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.add("hot"); });
for(const ev of ["dragleave","drop"])
  intake.addEventListener(ev, (e) => { e.preventDefault(); intake.classList.remove("hot"); });
intake.addEventListener("drop", (e) => stage([...e.dataTransfer.files]));
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

/* ---------- staging ---------- */
/* A dropped file is held, not sent: the controls below are read when Transcribe
   is pressed, so a recording survives a change of mind about the model, the
   language or the speaker count. Dropping used to start the job on the spot,
   which made a wrong setting cost a re-upload. */
const staged = [];

function fmtSize(bytes){
  const units = ["B", "kB", "MB", "GB"];
  let n = Number(bytes) || 0, i = 0;
  while(n >= 1024 && i < units.length - 1){ n /= 1024; i++; }
  // Whole units up to kB, one decimal from MB up: "312 kB", "1.4 GB".
  return `${i >= 2 ? n.toFixed(1) : Math.round(n)} ${units[i]}`;
}

function startLabel(count){
  return count > 1 ? `Transcribe ${count} files` : "Transcribe";
}

/* Says out loud what the button does, so the controls above it read as input to
   the press rather than as something already applied. */
function stageNote(count){
  if(!count) return "Drop a recording above to get started.";
  const what = count === 1 ? "1 file" : `${count} files`;
  return `${what} staged \u00b7 the controls above are read when you press Transcribe.`;
}

/* One row per staged file. Built with createElement and textContent: a filename
   is whatever the operator's filesystem said, and this path never touches
   innerHTML, so there is nothing here to escape. */
function renderStaged(){
  const box = el("staged");
  box.textContent = "";
  box.classList.toggle("locked", staged.length === 0);
  el("start").disabled = staged.length === 0;
  el("start").textContent = startLabel(staged.length);
  el("run-note").textContent = stageNote(staged.length);
  staged.forEach((file, index) => {
    const row = document.createElement("div");
    row.className = "staged-row";
    const name = document.createElement("span");
    name.className = "staged-name";
    name.textContent = file.name;
    name.title = file.name;
    const size = document.createElement("span");
    size.className = "staged-size";
    size.textContent = fmtSize(file.size);
    const drop = document.createElement("button");
    drop.className = "ghost staged-x";
    drop.type = "button";
    drop.textContent = "Remove";
    drop.dataset.unstage = String(index);
    drop.title = `Remove ${file.name}`;
    // Five buttons all called "Remove" are five identical names to a screen
    // reader; the label is what tells them apart.
    drop.setAttribute("aria-label", `Remove ${file.name}`);
    row.append(name, size, drop);
    box.append(row);
  });
}

function stage(files){
  for(const file of files){
    if(MAX_MB && file.size > MAX_MB * 1024 * 1024){
      alert(`${file.name} is larger than the ${MAX_MB} MB limit.`);
      continue;
    }
    staged.push(file);
  }
  renderStaged();
}

function unstage(index){
  staged.splice(index, 1);
  renderStaged();
}

el("staged").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-unstage]");
  if(button) unstage(Number(button.dataset.unstage));
});

el("start").addEventListener("click", startJobs);

/* Upload everything staged under the controls as they are right now. Reached
   only from the button: nothing else sends a file. */
async function startJobs(){
  if(!staged.length) return;
  const files = staged.splice(0, staged.length);
  renderStaged();
  await send(files);
}

/* The upload itself. XHR rather than api(), because fetch cannot report upload
   progress, and an hour of audio over the LAN is minutes of a page that
   otherwise says nothing. Same header, same 401 handling; the result is shaped
   like a fetch Response so send() reads it the same way. */
function postJob(body, onProgress){
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    const token = TOKENS.get();
    if(token) xhr.setRequestHeader("x-token", token);
    xhr.upload.onprogress = (e) => { if(e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onerror = () => reject(new Error("the connection failed"));
    xhr.onload = () => {
      if(xhr.status === 401){ lockPage(); reject(new Error("unauthorised")); return; }
      resolve({ok: xhr.status >= 200 && xhr.status < 300, statusText: xhr.statusText,
               json: async () => JSON.parse(xhr.responseText)});
    };
    xhr.send(body);
  });
}

/* What is on the wire right now. The staged list is emptied the moment
   Transcribe is pressed, so without this the page shows nothing at all between
   the press and the server accepting the file. */
function showUpload(file, fraction, waiting){
  const box = el("uploading");
  if(!file){ box.classList.add("locked"); return; }
  box.classList.remove("locked");
  el("up-name").textContent = file.name;
  el("up-name").title = file.name;
  // All bytes sent is not accepted yet: the server still has to store it.
  el("up-state").textContent = (fraction >= 1 ? "Handing over"
    : `Uploading ${Math.round(fraction * 100)}% of ${fmtSize(file.size)}`)
    + (waiting ? ` · ${waiting} more to send` : "");
  el("up-bar").value = fraction;
}

async function send(files){
  for(const [index, f] of files.entries()){
    /* Set once the server has taken the file. A failure after that — the poll
       that follows an upload — must not put it back: the job is already queued,
       and retrying would run the same audio twice. */
    let accepted = false;
    const waiting = files.length - index - 1;
    showUpload(f, 0, waiting);
    try{
      const fd = currentSettings();
      fd.append("file", f);
      const r = await postJob(fd, (fraction) => showUpload(f, fraction, waiting));
      if(!r.ok){
        let detail = r.statusText;
        try{ detail = (await r.json()).detail || detail; }catch{}
        throw new Error(detail);
      }
      await r.json();
      accepted = true;
      await tick();
    }catch(err){
      if(accepted) continue;
      if(err.message !== "unauthorised")
        alert(`Upload failed for ${f.name}: ${err.message}`);
      /* Put it back — including after a 401, where the gate swallows the
         message. The file is still in the picker, but re-picking an hour of
         audio because the network blipped is how a page stops being used.
         push(), not unshift(): a run of failures comes back in the order it was
         staged, so pressing Transcribe again retries them in that order. */
      staged.push(f);
      renderStaged();
    }
  }
  showUpload(null);
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
try{ el("follow").checked = sessionStorage.getItem(FOLLOW_KEY) !== "0"; }catch{}
el("follow").addEventListener("change", () => setFollow(el("follow").checked));

/* ---------- rendering ---------- */
const TICKS = 40;
const known = new Map();
/* One entry per card. The transcript keeps its DOM between polls so a running
   job can be read while it grows: replacing the whole card every 1.2s (what
   this used to do) threw away the scroll position and any selection. */
const views = new Map();

/* The one key for that map. Every lookup has to agree on it: createCard()
   stored by the bare job id while render() and tick() asked for "job-" + id, so
   every lookup missed, render() built a second card per progress step — meter,
   transcript and all — and the full transcript came back over the wire on every
   tick. One function means the next call site cannot pick a different spelling. */
function viewKey(id){
  return `job-${id}`;
}

/* The meter is 40 cells, and it is repainted on every poll: the cells are built
   once with the card and only re-classed afterwards, so a running job does not
   hand the collector 40 elements per tick. */
function meter(node, job){
  const lit = Math.round((job.progress || 0) * TICKS);
  const cls = job.state === "done" ? "done" : job.state === "error" ? "fail" : "lit";
  if(node.children.length !== TICKS){
    node.textContent = "";
    for(let i=0;i<TICKS;i++) node.append(document.createElement("i"));
  }
  [...node.children].forEach((cell, i) => { cell.className = i < lit ? cls : ""; });
}

function fmtTime(s){
  s = Math.round(s);
  const m = Math.floor(s/60);
  return m ? m + "m " + String(s%60).padStart(2,"0") + "s" : s + "s";
}

/* Everything here comes from the server, the model list or a job id, and the
   result is assigned with textContent. It returns text, not markup: do not
   hand it to innerHTML. */
function statusLine(job){
  if(job.state === "queued")  return "Waiting for the GPU";
  if(job.state === "loading") return "Loading " + (job.opts.model || "");
  if(job.state === "running"){
    // The diarization pass has no useful ETA of its own, and the meter is in
    // its last tenth by then, so report the phase rather than a wrong guess.
    if(job.phase === "diarizing") return job.message || "Identifying speakers";
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
      + " \u00b7 " + (job.language || "?") + " \u00b7 " + job.segment_count + " segments";
  }
  if(job.state === "cancelled") return "Cancelled";
  return job.message || "";
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
  try{ sessionStorage.setItem(FOLLOW_KEY, on ? "1" : "0"); }catch{}
  for(const view of views.values()){
    setPaused(view, false);
    if(on && view.live) stick(view);   // catch up now, not at the next segment
  }
}

function createCard(job){
  const node = document.createElement("div");
  node.className = "job";
  node.id = viewKey(job.id);
  /* Every part is built as a node rather than parsed from a markup string, so
     nothing a job carries — its filename, its message — can be read as markup
     in the first place, and no call site has to remember to escape it. The
     parts are kept on the view, which also makes render() a matter of setting
     text and classes instead of four querySelector calls per poll. */
  const view = {node, shown: 0, paused: false, live: false, labeled: false};
  view.name = document.createElement("span");
  view.name.className = "job-name";
  view.meta = document.createElement("span");
  view.meta.className = "job-meta";
  view.head = document.createElement("div");
  view.head.className = "job-head";
  view.head.append(view.name, view.meta);
  view.meter = document.createElement("div");
  view.meter.className = "meter";
  view.status = document.createElement("p");
  view.status.className = "status";
  view.transcript = document.createElement("div");
  view.transcript.className = "transcript";
  view.actions = document.createElement("div");
  view.actions.className = "actions";
  const body = document.createElement("div");
  body.className = "job-body";
  body.append(view.status, view.transcript, view.actions);
  node.append(view.head, view.meter, body);
  el("jobs").prepend(node);

  /* Scrolling away from the newest line pauses; coming back re-arms. The
     Follow switch stays the authoritative off switch. */
  view.transcript.addEventListener("scroll", () => {
    const atBottom = view.transcript.scrollHeight - view.transcript.scrollTop
                     - view.transcript.clientHeight <= 8;
    if(atBottom === view.paused) setPaused(view, !atBottom);
  });
  views.set(viewKey(job.id), view);
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
  /* The speaker column only takes room once there is something in it: a 9ch
     gap between every timestamp and its text read as broken layout. */
  view.transcript.classList.toggle("labeled", !!view.labeled);
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

/* A button per action. Built rather than written: a job id lands in a data
   attribute and the label is text, so a filename or an error message never has
   to be escaped to be safe here. */
function actions(node, job){
  node.textContent = "";
  const add = (label, data, ghost) => {
    const button = document.createElement("button");
    button.textContent = label;
    Object.assign(button.dataset, data);
    if(ghost) button.className = "ghost";
    node.append(button);
  };
  if(job.state === "done"){
    add("Copy text", {copy: job.id});
    for(const [fmt, label] of [["txt", "Save .txt"],
                               ["timestamped", "Save timestamped"],
                               ["srt", "Save .srt"],
                               ["vtt", "Save .vtt"],
                               ["json", "Save .json"]])
      add(label, {dl: fmt, id: job.id});
  }
  if(RETRY_OK && job.can_retry)
    add("Retry with these settings", {retry: job.id});
  add(["queued","loading","running"].includes(job.state) ? "Cancel" : "Remove",
      {del: job.id}, true);
}

function render(job){
  let view = views.get(viewKey(job.id));
  if(!view || !view.node.isConnected) view = createCard(job);

  view.name.textContent = job.filename || "";
  view.name.title = job.filename || "";
  view.meta.textContent = jobTags(job);
  meter(view.meter, job);
  view.status.className = job.state === "error" ? "status err" : "status";
  view.status.textContent = statusLine(job);
  /* Rebuilt only when the set of buttons changes. A running job renders every
     poll, and a fresh Cancel under the pointer each time ate any click whose
     press and release straddled a poll, and threw keyboard focus off it. */
  const shape = job.state + ":" + !!(RETRY_OK && job.can_retry);
  if(view.actionShape !== shape){
    view.actionShape = shape;
    actions(view.actions, job);
  }

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
  if(star) { try { name = decodeURIComponent(star[1]); } catch{} }
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
        try{ detail = (await r.json()).detail || detail; }catch{}
        alert("Retry failed: " + detail);
        b.disabled = false;
      }
      await tick();
    }
    if(b.dataset.del){
      await api("/api/jobs/" + encodeURIComponent(b.dataset.del), {method:"DELETE"});
      await tick();
    }
  }catch{ /* 401 already handled by api() */ }
});

/* ---------- poll ---------- */
/* One poll at a time. tick() is called by the interval and again after every
   upload, retry and delete, and a poll is several awaits long. Two in flight
   both read a card's row count before either appended, so both asked for the
   same tail: the second to land drew rows the first already had, or its shorter
   total read as a transcript going backwards and wiped the card down to a
   tail. A call that arrives mid-poll asks for one more pass and waits for it,
   so "tick() after a DELETE" still means a list read after the DELETE. */
let polling = null, pollAgain = false;

function tick(){
  if(polling){ pollAgain = true; return polling; }
  polling = (async () => {
    try{
      do{ pollAgain = false; await poll(); }while(pollAgain);
    }finally{ polling = null; }
  })();
  return polling;
}

async function poll(){
  if(!unlocked) return;
  try{
    const {jobs} = await (await api("/api/jobs")).json();
    const live = jobs.map(j => j.id);
    el("empty").classList.toggle("locked", jobs.length > 0);

    for(const stale of [...known.keys()].filter(id => !live.includes(id))){
      const n = document.getElementById(viewKey(stale));
      if(n) n.remove();
      known.delete(stale);
      views.delete(viewKey(stale));
    }

    /* Oldest first, because a new card is prepended: the server lists newest
       first, and walking that order stacked a reloaded page upside down while
       a job added afterwards still went on top. */
    for(const summary of [...jobs].reverse()){
      /* Phase and message too: fetching the diarization models moves neither
         the meter nor the segment count, and the card would go on promising an
         ETA for the whole download. */
      const sig = [summary.state, summary.phase, summary.message,
                   summary.progress, summary.segment_count].join(":");
      if(known.get(summary.id) === sig) continue;

      /* Ask for only what this card has not seen. A card that lost its rows
         (the job's segments went backwards) starts from zero again. */
      const view = views.get(viewKey(summary.id));
      let since = view ? view.shown : 0;
      if(view && summary.segment_count < view.shown){
        view.transcript.textContent = "";
        view.shown = 0;
        since = 0;
      }
      const url = "/api/jobs/" + encodeURIComponent(summary.id);
      const detail = await api(url + "?since=" + since);
      /* A job can vanish between the list and this read (evicted from memory, or
         a blip answered with an error body). Rendering the body as a job would
         key a card on undefined, which the sweep below — walking real ids —
         could then never remove: a phantom that survives until the page is
         reloaded. Skip it; `known` is only written after a render, so the next
         poll asks again instead of taking this change as seen. */
      if(!detail.ok) continue;
      let full = await detail.json();
      if(!full || full.id !== summary.id) continue;
      /* Two things a tail cannot repair, both answered by one full refetch:
         labels arriving in one batch when diarization finishes, which rows
         already drawn do not have, and a transcript that shrank between the
         list and this read, whose tail from `since` is empty. */
      if(view && view.shown > 0 && (full.speaker_labels !== view.labeled
                                    || full.segment_count < since)){
        view.transcript.textContent = "";
        view.shown = 0;
        const again = await api(url + "?since=0");
        if(!again.ok) continue;
        full = await again.json();
        if(!full || full.id !== summary.id) continue;
      }
      render(full);
      known.set(summary.id, sig);
    }
  }catch{ /* server blip or 401; next tick retries */ }
}

async function boot(){
  await refreshStatus();
  await tick();
}

boot();
setInterval(tick, 1200);
setInterval(refreshStatus, 5000);
