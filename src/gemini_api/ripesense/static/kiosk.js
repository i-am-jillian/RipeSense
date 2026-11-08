// RipeSense kiosk: polls /api/state every 400 ms and renders idle, scanning, result or error.
const $ = (id) => document.getElementById(id);
const VIEWS = ["idle", "scanning", "result", "error"];
const STEPS = { door: ["i-door", "Checking door"], camera: ["i-camera", "Taking a photo"], spectral: ["i-spectrum", "Reading spectrum"],
  gas: ["i-gas", "Sniffing gases"], model: ["i-spark", "Thinking"] };
const NM_COLORS = ["#7b4dff", "#3d6bff", "#1fb6ff", "#20c997", "#8bd450", "#ffd23f", "#ff8c2a", "#e8412c"];
const STAGE_POS = { unripe: 0, ripe: 1 / 3, overripe: 2 / 3, spoiled: 1 };
const RING = 2 * Math.PI * 68, CONF = 2 * Math.PI * 18;
let shown = null, lastScan = null, lastToken = null, qrFor = null, lastResult = null;

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return r.ok ? r.json() : null;
}
const cap = (t) => (t ? t[0].toUpperCase() + t.slice(1) : "");
const fmt = (x) => String(Math.round(x));
function show(v) { if (shown !== v) { VIEWS.forEach((n) => ($(n).hidden = n !== v)); shown = v; } }
function setIcon(svgEl, id) { svgEl.firstElementChild.setAttribute("href", "#" + id); }

// ------------------------------------------------------------------ idle
function renderIdle(s) {
  const warm = s.gas_warmup_left_s > 0;
  $("rGas").className = "rchip" + (warm ? " warm" : "");
  $("rGas").querySelector("em").textContent = warm ? `Gas warming ${Math.ceil(s.gas_warmup_left_s / 60)} min` : "Gas";
  const r = lastResult;
  $("last").textContent = r ? `Last scan · ${cap(r.fruit)} · ${r.prediction.stage}` + (r.days != null && r.prediction.stage !== "spoiled" ? ` · ${r.days} days` : "") : "";
}

// ------------------------------------------------------------------ scanning
function renderScan(s) {
  const live = s.live || {};
  const [icon, label] = STEPS[s.step] || STEPS.door;
  setIcon($("stepIcon"), icon);
  $("stepLabel").textContent = label;
  $("arc").style.strokeDasharray = RING;
  $("arc").style.strokeDashoffset = RING * (1 - (s.progress || 0));
  $("ringNum").textContent = s.step === "gas" && live.seconds_left != null ? `${live.seconds_left}s` : `${Math.round((s.progress || 0) * 100)}%`;

  const order = ["camera", "spectral", "gas"];
  const idx = order.indexOf(s.step);
  ["cCamera", "cSpectral", "cGas"].forEach((id, k) => $(id).classList.toggle("active", k === idx));

  if (live.token !== lastToken) { lastToken = live.token; $("livePhoto").removeAttribute("src"); $("spec").replaceChildren(); $("gas").replaceChildren(); }
  if (live.preview && !$("livePhoto").getAttribute("src")) $("livePhoto").src = `/api/live/preview.jpg?t=${live.token}`;
  $("cCamera").classList.toggle("has", !!live.preview);
  $("camSub").textContent = live.preview ? "Fruit found" : s.step === "camera" ? "Lights on…" : "Waiting";

  if (live.spectrum) {
    $("cSpectral").classList.add("has");
    const max = Math.max(...live.spectrum, 0.05);
    const bars = live.spectrum.map((v, i) => {
      const h = Math.max(2, (v / max) * 44);
      return `<rect x="${4 + i * 14.4}" y="${48 - h}" width="10" height="${h}" rx="2" fill="${NM_COLORS[i]}"/>`;
    }).join("");
    $("spec").innerHTML = bars;
    $("specSub").textContent = "8 bands · 415–680 nm";
  } else { $("cSpectral").classList.remove("has"); $("specSub").textContent = s.step === "spectral" ? "Measuring…" : "415–680 nm"; }

  const g = live.gas || [];
  $("cGas").classList.toggle("has", g.length > 1);
  if (g.length > 1) {
    const lo = Math.min(1, ...g), hi = Math.max(1.02, ...g);
    const pts = g.map((v, i) => `${(i / (g.length - 1)) * 116 + 2},${48 - ((v - lo) / (hi - lo)) * 42}`).join(" ");
    const base = 48 - ((1 - lo) / (hi - lo)) * 42;
    $("gas").innerHTML = `<line x1="0" x2="120" y1="${base}" y2="${base}" stroke="#3a3f46" stroke-dasharray="3 3"/>` +
      `<polyline points="${pts}" fill="none" stroke="#ffb020" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`;
    const rise = live.gas_rise == null ? 0 : live.gas_rise * 100;
    $("gasSub").textContent = `${rise >= 0 ? "+" : ""}${rise.toFixed(1)}% vs room air`;
  } else { $("gasSub").textContent = s.step === "gas" ? "Collecting…" : "Waiting"; }
}

// ------------------------------------------------------------------ result
function renderResult(s) {
  const r = s.result, p = r.prediction, stage = p.uncertain ? "uncertain" : p.stage;
  lastResult = r;
  const col = p.uncertain ? "var(--bad)" : `var(--${p.stage})`;
  if (s.scan_id !== lastScan) {
    lastScan = s.scan_id; qrFor = null;
    $("thumb").src = `/api/scans/${s.scan_id}/thumb.jpg`;
  }
  $("stageChip").textContent = stage;
  $("stageChip").style.background = col;
  $("fruitName").textContent = cap(r.fruit) + (p.mode === "heuristic" ? " · rough estimate" : "");
  $("headline").textContent = r.headline;
  document.querySelector(".photo").style.borderBottomColor = col;

  const ok = Number.isFinite(p.days_low) && p.stage !== "spoiled";
  $("daysNum").textContent = p.stage === "spoiled" ? "0" : ok ? (fmt(p.days_low) === fmt(p.days_high) ? fmt(p.days_mid) : `${fmt(p.days_low)}–${fmt(p.days_high)}`) : "—";
  const conf = Math.max(0, Math.min(1, p.confidence || 0));
  $("confArc").style.strokeDasharray = CONF;
  $("confArc").style.strokeDashoffset = CONF * (1 - conf);
  $("confArc").style.stroke = conf >= 0.75 ? "var(--ripe)" : conf >= 0.6 ? "var(--accent)" : "var(--bad)";
  $("confNum").textContent = `${Math.round(conf * 100)}`;
  const pos = r.gauge != null ? r.gauge : STAGE_POS[p.stage] || 0.33;
  $("gMark").style.left = `${Math.min(97, Math.max(3, pos * 100))}%`;

  const rd = r.readings || {}, n = (x) => x != null && Number.isFinite(x);
  const vals = {
    camera: [n(rd.brown_pct) ? `${Math.round(rd.brown_pct)}% brown` : n(rd.length_mm) ? `${Math.round(rd.length_mm)} mm long` : null],
    spectral: [n(rd.ripen_idx) ? `index ${rd.ripen_idx >= 0 ? "+" : ""}${rd.ripen_idx.toFixed(2)}` : null],
    gas: [n(rd.gas_pct) ? `${rd.gas_pct >= 0 ? "+" : ""}${rd.gas_pct.toFixed(1)}% vs air` : "not used"],
  };
  document.querySelectorAll("#agree > div").forEach((el) => {
    const v = p.agreement[el.dataset.m];
    el.className = v === undefined ? "off" : v === p.stage ? "yes" : "no";
    setIcon(el.querySelector(".s"), v === undefined || v === p.stage ? "i-check" : "i-x");
    const parts = vals[el.dataset.m].filter(Boolean);
    el.querySelector(".av").textContent = v !== undefined && v !== p.stage ? `says ${v}` : parts.join(" · ") || "—";
  });

  const b = $("bottom");
  b.replaceChildren();
  if (s.mode === "store" && r.price != null) {
    b.append(span("price", `${r.currency}${r.price.toFixed(2)}`));
    if (r.markdown) { b.append(span("was", `${r.currency}${r.list_price.toFixed(2)}`), span("off", `−${r.markdown[0]}–${r.markdown[1]}%`)); }
    b.append(span("act", r.action));
  } else {
    if (r.tip_source === "gemini") b.insertAdjacentHTML("beforeend", '<svg class="gem"><use href="#i-spark"/></svg>');
    b.append(span("", r.tip || ""));
  }

  const card = $("recipeCard"), st = r.recipe_status;
  card.classList.toggle("ready", st === "ready");
  $("qrWrap").hidden = st !== "ready"; $("recipeLbl").hidden = st !== "ready"; $("recipeWait").hidden = st === "ready";
  if (st === "ready") {
    if (qrFor !== s.scan_id) { qrFor = s.scan_id; $("qrImg").src = $("qrBigImg").src = `/api/scans/${s.scan_id}/qr.svg`; }
    $("recipeTitle").textContent = r.recipe_title || "";
    $("qrBigTitle").textContent = r.recipe_title || "Recipe";
    $("qrBigSrc").textContent = r.recipe_source === "built-in" ? "Built-in recipe" : r.recipe_source ? `Written by ${r.recipe_source}` : "";
  } else {
    $("recipeWait").lastElementChild.textContent = st === "pending" ? "Gemini is writing a recipe…" : p.stage === "spoiled" ? "No recipe: compost it" : "No recipe";
    $("recipeWait").firstElementChild.hidden = st !== "pending";
  }
}
function span(cls, text) { const e = document.createElement("span"); if (cls) e.className = cls; e.textContent = text; return e; }

// ------------------------------------------------------------------ loop
function render(s) {
  document.querySelectorAll(".seg button").forEach((bt) => bt.classList.toggle("on", bt.dataset.mode === s.mode));
  $("untrained").hidden = s.predictor !== "heuristic";
  $("rescanBtn").hidden = !(s.status === "result" || s.status === "error");
  if (s.status !== "result") $("qrBig").hidden = true;
  if (s.status === "scanning") { show("scanning"); renderScan(s); }
  else if (s.status === "result" && s.result) { show("result"); renderResult(s); }
  else if (s.status === "error") {
    show("error"); $("errMsg").textContent = s.message;
    setIcon($("errIcon"), /door/i.test(s.message) ? "i-door" : "i-alert");
  } else { show("idle"); renderIdle(s); }
}
async function poll() {
  try { const s = await api("/api/state"); if (s) render(s); } catch (e) { /* server restarting */ }
  setTimeout(poll, 400);
}
const scan = () => api("/api/scan", {});
$("tapBtn").onclick = scan; $("retryBtn").onclick = scan; $("rescanBtn").onclick = scan;
document.querySelectorAll(".seg button").forEach((bt) => (bt.onclick = () => api("/api/mode", { mode: bt.dataset.mode })));
$("recipeCard").onclick = () => { if ($("recipeCard").classList.contains("ready")) $("qrBig").hidden = false; };
$("qrBig").onclick = () => ($("qrBig").hidden = true);
poll();
