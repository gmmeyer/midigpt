"use strict";
/*
 * midigpt — client-side. A MusicGPT (model.onnx) generates event tokens
 * autoregressively via onnxruntime-web (WebGPU, WASM fallback); the JS decode
 * loop mirrors midigpt/tokenizer.py exactly (sustain already folded at train
 * time, so here NOTE_ON/OFF + TIME_SHIFT + VELOCITY just rebuild notes). Notes
 * play through a Tone.js polyphonic synth and scroll by on a piano roll.
 */

// config.json ships in the repo (served next to this page). model.onnx is too
// big for GitHub, so on a deployed page it's fetched from Hugging Face; on
// localhost it's read from ./model.onnx (regenerate with export_onnx.py).
const HF_URL = "https://huggingface.co/gmmeyer/midigpt/resolve/main";
const LOCAL = ["localhost", "127.0.0.1", ""].includes(location.hostname);
const MODEL_URL = (LOCAL ? "." : HF_URL) + "/model.onnx";

let cfg, L, session, backend = "—";
let generating = false, stopFlag = false;
let lastNotes = [], lastIds = [];

const $ = (id) => document.getElementById(id);
const setStatus = (h) => { $("status").innerHTML = h; };

async function init() {
  cfg = await (await fetch("./config.json")).json();
  L = cfg.layout;

  // composer dropdown, alphabetical
  const sel = $("composer");
  Object.keys(cfg.composers).sort().forEach((name) => {
    const o = document.createElement("option");
    o.value = cfg.composers[name]; o.textContent = name;
    sel.appendChild(o);
  });

  setStatus("loading model (~106&nbsp;MB, one-time)…");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  const warm = [L.COMPOSER_OFF, cfg.specials.BOS, L.VELOCITY_OFF, L.NOTE_ON_OFF];
  // Try WebGPU, warming up with a real inference so a run-time failure (not just
  // a load-time one) falls back to WASM, which always works.
  try {
    session = await ort.InferenceSession.create(MODEL_URL,
      { executionProviders: ["webgpu"] });
    await runLast(warm);
    backend = "WebGPU";
  } catch (eGpu) {
    try {
      session = await ort.InferenceSession.create(MODEL_URL,
        { executionProviders: ["wasm"] });
      await runLast(warm);
      backend = "WASM";
    } catch (e) { setStatus("failed to load model: " + e.message); return; }
  }

  $("pill").textContent = backend;
  $("generate").disabled = false;
  setStatus("ready — pick a composer and hit Generate");
  loadInstrument();   // preload piano buffers in the background (no gesture needed)
}

// ---- token classification (mirrors vocab.py) ----
const kindOf = (t) => {
  if (t === cfg.specials.EOS) return "eos";
  if (t >= L.NOTE_ON_OFF && t < L.NOTE_OFF_OFF) return "note_on";
  if (t >= L.NOTE_OFF_OFF && t < L.TIME_SHIFT_OFF) return "note_off";
  if (t >= L.TIME_SHIFT_OFF && t < L.VELOCITY_OFF) return "shift";
  if (t >= L.VELOCITY_OFF && t < cfg.vocab_size) return "velocity";
  return "other";
};
const pitchOnOf = (t) => t - L.NOTE_ON_OFF + cfg.pitch_min;
const pitchOffOf = (t) => t - L.NOTE_OFF_OFF + cfg.pitch_min;
const shiftOf = (t) => t - L.TIME_SHIFT_OFF + 1;
const velBinOf = (t) => t - L.VELOCITY_OFF;
const velBinToMidi = (b) => b * 4 + 2;

// ---- decode a full id stream to notes (mirrors tokenizer.decode) ----
function decodeNotes(ids) {
  let cur = 0, curVel = velBinToMidi(Math.floor(cfg.n_velocity / 2));
  const active = new Map();            // pitch -> {start, vel}
  const notes = [];
  const close = (p, end) => {
    const a = active.get(p); active.delete(p);
    notes.push({ start: a.start, dur: Math.max(end - a.start, 1), pitch: p, vel: a.vel });
  };
  for (const t of ids) {
    const k = kindOf(t);
    if (k === "shift") cur += shiftOf(t);
    else if (k === "velocity") curVel = velBinToMidi(velBinOf(t));
    else if (k === "note_on") { const p = pitchOnOf(t); if (active.has(p)) close(p, cur); active.set(p, { start: cur, vel: curVel }); }
    else if (k === "note_off") { const p = pitchOffOf(t); if (active.has(p)) close(p, cur); }
    else if (k === "eos") break;
  }
  for (const p of [...active.keys()].sort((a, b) => a - b)) close(p, cur);
  notes.sort((a, b) => a.start - b.start);
  return notes;
}

// ---- sampling ----
function sampleLogits(logits, temperature, topP) {
  // Only event tokens (NOTE_ON..VELOCITY) are legal. EOS is deliberately excluded:
  // it fires early ~10-20% of the time and would truncate the piece to a stub, so
  // the demo generates the full requested length and keeps improvising instead.
  const legalLo = L.NOTE_ON_OFF;
  const V = cfg.vocab_size;
  const cand = [];
  let mx = -Infinity;
  for (let i = legalLo; i < V; i++) {
    const v = logits[i] / Math.max(temperature, 1e-6);
    cand.push([i, v]); if (v > mx) mx = v;
  }
  let sum = 0;
  for (const c of cand) { c[1] = Math.exp(c[1] - mx); sum += c[1]; }
  for (const c of cand) c[1] /= sum;
  cand.sort((a, b) => b[1] - a[1]);
  // nucleus
  let acc = 0, cut = cand.length;
  for (let i = 0; i < cand.length; i++) { acc += cand[i][1]; if (acc >= topP) { cut = i + 1; break; } }
  const keep = cand.slice(0, cut);
  let ks = 0; for (const c of keep) ks += c[1];
  let r = Math.random() * ks, a2 = 0;
  for (const c of keep) { a2 += c[1]; if (r <= a2) return c[0]; }
  return keep[keep.length - 1][0];
}

async function runLast(window) {
  const w = window.slice(-cfg.web_context_cap);
  const t = new ort.Tensor("int64", BigInt64Array.from(w, (v) => BigInt(v)), [1, w.length]);
  return (await session.run({ idx: t })).logits.data;   // Float32Array, length vocab
}

async function generate() {
  if (generating) return;
  generating = true; stopFlag = true; if (window.Tone) Tone.Transport.stop();
  stopFlag = false;
  $("generate").disabled = true; $("stop").disabled = false;

  const composerTok = L.COMPOSER_OFF + parseInt($("composer").value, 10);
  const temperature = parseFloat($("temp").value);
  const topP = parseFloat($("topp").value);
  const maxTokens = parseInt($("length").value, 10);

  const ctx = [composerTok, cfg.specials.BOS];
  const t0 = performance.now();
  for (let n = 0; n < maxTokens; n++) {
    if (stopFlag) break;
    const logits = await runLast(ctx);
    const tok = sampleLogits(logits, temperature, topP);
    ctx.push(tok);
    if (tok === cfg.specials.EOS) break;
    if (n % 16 === 0) {
      setStatus(`generating… ${n}/${maxTokens} tokens · ${backend} · ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      drawRoll(decodeNotes(ctx));
      await new Promise((r) => setTimeout(r, 0));   // yield so UI updates
    }
  }
  lastIds = ctx;
  lastNotes = decodeNotes(ctx);
  drawRoll(lastNotes);
  const dur = lastNotes.length ? Math.max(...lastNotes.map((n) => (n.start + n.dur))) * cfg.time_step_ms / 1000 : 0;
  setStatus(`done — ${lastNotes.length} notes, ${dur.toFixed(1)}s. Press Play.`);
  $("generate").disabled = false; $("stop").disabled = true;
  $("play").disabled = lastNotes.length === 0;
  $("download").disabled = lastNotes.length === 0;
  generating = false;
}

// ---- playback via Tone.js ----
// Real piano samples (Salamander grand) — a synth makes dense piano writing sound
// harsh/dissonant; sampled piano sounds like the notes actually mean. Falls back
// to a warm synth if the sample CDN can't be reached.
const SALAMANDER = "https://tonejs.github.io/audio/salamander/";
const PIANO_URLS = (() => {
  const urls = {};
  for (let oct = 0; oct <= 8; oct++) {
    for (const [note, file] of [["A", "A"], ["C", "C"], ["D#", "Ds"], ["F#", "Fs"]]) {
      if ((oct === 0 && note !== "A") || oct === 8 && note !== "C") continue;
      urls[note + oct] = file + oct + ".mp3";
    }
  }
  return urls;                       // A0, C1, D#1, F#1, A1, … C8
})();

let instrument = null;
let instrumentPromise = null;
// Load the piano buffers in the background — this needs NO audio gesture (fetch
// only), so calling it right after the model loads makes Play instant. Only
// Tone.start() in play() needs the user's click.
function loadInstrument() {
  if (instrumentPromise) return instrumentPromise;
  instrumentPromise = (async () => {
    try {
      const piano = new Tone.Sampler({ urls: PIANO_URLS, baseUrl: SALAMANDER,
        release: 1.0 }).toDestination();
      await Promise.race([
        Tone.loaded(),
        new Promise((_, rej) => setTimeout(() => rej(new Error("sample timeout")), 15000)),
      ]);
      instrument = piano;
    } catch (e) {
      const synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: "triangle" },
        envelope: { attack: 0.004, decay: 0.3, sustain: 0.15, release: 1.0 },
        volume: -8,
      }).toDestination();
      synth.maxPolyphony = 64;
      instrument = synth;
    }
    return instrument;
  })();
  return instrumentPromise;
}

async function play() {
  if (!lastNotes.length) return;
  await Tone.start();                 // resumes the AudioContext (needs the click)
  if (!instrument) setStatus("loading piano samples…");
  const inst = await loadInstrument();
  Tone.Transport.stop(); Tone.Transport.cancel();
  const s = cfg.time_step_ms / 1000;
  for (const nt of lastNotes) {
    Tone.Transport.schedule((time) => {
      inst.triggerAttackRelease(Tone.Frequency(nt.pitch, "midi"), Math.max(nt.dur * s, 0.05),
        time, Math.min(nt.vel / 127, 1));
    }, nt.start * s);
  }
  const end = Math.max(...lastNotes.map((n) => n.start + n.dur)) * s + 1.5;
  Tone.Transport.scheduleOnce(() => { Tone.Transport.stop(); }, end);
  setStatus(`playing — ${lastNotes.length} notes`);
  Tone.Transport.start();
  animatePlayhead(end);
}

// ---- piano-roll rendering ----
function drawRoll(notes, playSec) {
  const cv = $("roll"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  g.fillStyle = "#0e1020"; g.fillRect(0, 0, W, H);
  if (!notes.length) return;
  const s = cfg.time_step_ms / 1000;
  const endGrid = Math.max(...notes.map((n) => n.start + n.dur), 1);
  const loP = cfg.pitch_min, hiP = cfg.pitch_min + cfg.n_pitch;
  const xOf = (grid) => (grid / endGrid) * (W - 8) + 4;
  const yOf = (p) => H - ((p - loP) / (hiP - loP)) * (H - 8) - 4;
  for (const n of notes) {
    const x = xOf(n.start), w = Math.max(xOf(n.start + n.dur) - x, 1.5);
    const y = yOf(n.pitch), hue = 200 + (n.pitch - loP) / cfg.n_pitch * 120;
    g.fillStyle = `hsl(${hue} 70% ${35 + n.vel / 4}%)`;
    g.fillRect(x, y - 2, w, 4);
  }
  if (playSec != null) {
    const px = xOf(playSec / s);
    g.strokeStyle = "#ffd34d"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke();
  }
}

function animatePlayhead(endSec) {
  const startT = Tone.now();
  const tick = () => {
    if (Tone.Transport.state !== "started") { drawRoll(lastNotes); return; }
    drawRoll(lastNotes, Tone.now() - startT);
    requestAnimationFrame(tick);
  };
  tick();
}

// ---- MIDI download (minimal SMF writer, tpq=500 qpm=120 => 1 tick = 1ms) ----
function downloadMidi() {
  if (!lastNotes.length) return;
  const bytes = buildMidi(lastNotes);
  const blob = new Blob([bytes], { type: "audio/midi" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "midigpt.mid"; a.click();
}

function buildMidi(notes) {
  const TPQ = 500, ticksPerGrid = cfg.time_step_ms;   // 10 ticks/grid at tpq500/qpm120
  const ev = [];
  for (const n of notes) {
    ev.push({ t: n.start * ticksPerGrid, on: true, p: n.pitch, v: n.vel });
    ev.push({ t: (n.start + n.dur) * ticksPerGrid, on: false, p: n.pitch, v: 0 });
  }
  ev.sort((a, b) => a.t - b.t || (a.on ? 1 : -1));
  const track = [];
  const vlq = (n) => { const b = [n & 0x7f]; n >>= 7; while (n) { b.unshift((n & 0x7f) | 0x80); n >>= 7; } return b; };
  // tempo 120bpm = 500000 us/qn
  track.push(...vlq(0), 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20);
  let last = 0;
  for (const e of ev) {
    track.push(...vlq(e.t - last)); last = e.t;
    track.push(e.on ? 0x90 : 0x80, e.p & 0x7f, e.v & 0x7f);
  }
  track.push(0x00, 0xff, 0x2f, 0x00);
  const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
  const head = [0x4d, 0x54, 0x68, 0x64, ...u32(6), 0x00, 0x00, 0x00, 0x01, (TPQ >> 8) & 255, TPQ & 255];
  const trkHead = [0x4d, 0x54, 0x72, 0x6b, ...u32(track.length)];
  return new Uint8Array([...head, ...trkHead, ...track]);
}

function stop() { stopFlag = true; if (window.Tone) Tone.Transport.stop(); }

$("generate").addEventListener("click", generate);
$("play").addEventListener("click", play);
$("stop").addEventListener("click", stop);
$("download").addEventListener("click", downloadMidi);
$("temp").addEventListener("input", (e) => { $("tempval").textContent = (+e.target.value).toFixed(2); });
$("topp").addEventListener("input", (e) => { $("toppval").textContent = (+e.target.value).toFixed(2); });

init();
