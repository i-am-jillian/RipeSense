// RipeSense insight panels: why-map, forecast + what-ifs, future preview,
// fruit profile + recipe, store optimizer, IR bruises. Vanilla JS, no deps.
const RS = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const ZONES = [["unripe", 0, 0.6, "Unripe"], ["ripe", 0.6, 1.25, "Ripe"], ["ripe_to_overripe", 1.25, 1.75, "Eat soon"],
                 ["overripe", 1.75, 2.5, "Overripe"], ["spoiled", 2.5, 3.2, "Spoiled"]];
  const zoneOf = q => (ZONES.find(z => q < z[2]) || ZONES[4])[3];
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const pct = v => Math.round(v * 100) + "%";
  const DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  function when(iso) {
    const d = new Date(iso), now = new Date();
    const days = Math.round((new Date(d.toDateString()) - new Date(now.toDateString())) / 864e5);
    const h = d.getHours(), part = h < 5 ? "night" : h < 12 ? "morning" : h < 17 ? "afternoon" : h < 21 ? "evening" : "night";
    if (days === 0) return part === "night" && h >= 21 ? "tonight" : "this " + part;
    return (days === 1 ? (part === "night" && h >= 21 ? "tomorrow" : "tomorrow") : DAY[d.getDay()]) + " " + part;
  }
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function tipLayer(box, svg, W, x0, x1, onX) {
    const tip = document.createElement("div"); tip.className = "tip"; box.appendChild(tip);
    const hair = el("line", {y1: 6, y2: 0, stroke: "#c3c2b7", "stroke-width": 1, opacity: 0}, svg);
    const hit = el("rect", {x: x0, y: 0, width: x1 - x0, height: "100%", fill: "transparent"}, svg);
    hit.addEventListener("pointermove", ev => {
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width;
      const res = onX(Math.min(x1, Math.max(x0, sx)));
      if (!res) return;
      hair.setAttribute("x1", res.x); hair.setAttribute("x2", res.x);
      hair.setAttribute("y2", res.yb); hair.setAttribute("opacity", .6);
      tip.innerHTML = res.html; tip.style.display = "block";
      const px = res.x * r.width / W;
      tip.style.left = Math.min(r.width - tip.offsetWidth - 2, Math.max(0, px + 10)) + "px"; tip.style.top = "4px";
    });
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; hair.setAttribute("opacity", 0); });
  }

  // ------------------------------------------------------------ forecast chart
  function forecastChart(box, f) {
    box.innerHTML = "";
    const W = 340, H = 190, L = 26, R = 62, T = 8, B = 22;
    const spoil = f.milestones.spoiled.days[2];
    const hz = Math.min(f.curve.days[f.curve.days.length - 1], Math.max(4, Math.ceil(spoil + 1)));
    const n = f.curve.days.findIndex(d => d > hz + 1e-9); const N = n < 0 ? f.curve.days.length : n;
    const X = d => L + (W - L - R) * d / hz, Y = q => T + (H - T - B) * (1 - q / 3.2);
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": `Ripeness forecast: ${zoneOf(f.q_now)} now, peak window ${when(f.peak_window.start)} to ${when(f.peak_window.end)}`});
    box.appendChild(svg);
    for (const [k, a, b, lab] of ZONES) {
      el("rect", {x: L, y: Y(b), width: W - L - R, height: Y(a) - Y(b), fill: `var(--z-${k})`, opacity: .13}, svg);
      el("text", {x: W - R + 6, y: (Y(a) + Y(b)) / 2 + 3, class: "zone"}, svg).textContent = lab;
    }
    const now = new Date(f.now);
    for (let d = 0; d <= hz; d += hz > 7 ? 2 : 1) {
      el("line", {x1: X(d), x2: X(d), y1: T, y2: H - B, stroke: "#2e3432", "stroke-width": 1}, svg);
      const dt = new Date(now.getTime() + d * 864e5);
      el("text", {x: X(d), y: H - 7, "text-anchor": "middle"}, svg).textContent = d === 0 ? "now" : DAY[dt.getDay()];
    }
    const pts = (arr) => f.curve.days.slice(0, N).map((d, i) => `${X(d).toFixed(1)},${Y(arr[i]).toFixed(1)}`);
    el("polygon", {points: pts(f.curve.p90).concat(pts(f.curve.p10).reverse()).join(" "), fill: "var(--series-1)", opacity: .25}, svg);
    el("polyline", {points: pts(f.curve.p50).join(" "), fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
    el("circle", {cx: X(0), cy: Y(f.curve.p50[0]), r: 4.5, fill: "var(--series-1)", stroke: "#222625", "stroke-width": 2}, svg);
    tipLayer(box, svg, W, L, W - R, sx => {
      const d = (sx - L) / (W - L - R) * hz;
      let i = 0; while (i < N - 1 && f.curve.days[i + 1] <= d + 0.125) i++;
      const dt = new Date(now.getTime() + f.curve.days[i] * 864e5);
      return {x: X(f.curve.days[i]), yb: H - B,
        html: `<b>${DAY[dt.getDay()]} ${dt.getHours()}:00</b> · ${zoneOf(f.curve.p50[i])}<br>` +
              `<span style="color:#8b9490">likely range: ${zoneOf(f.curve.p10[i])} – ${zoneOf(f.curve.p90[i])}</span>`};
    });
  }

  // ------------------------------------------------------------- price chart
  function priceChart(box, o, base) {
    box.innerHTML = "";
    const W = 340, H = 140, L = 34, R = 10, T = 10, B = 22;
    const c = o.curve, vmax = Math.max(...c.map(p => p.value), base * 0.35, 0.01), vmin = Math.min(0, ...c.map(p => p.value));
    const X = p => L + (W - L - R) * p / 70, Y = v => T + (H - T - B) * (1 - (v - vmin) / (vmax - vmin));
    const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Expected recovery by discount"});
    box.appendChild(svg);
    for (const t of [0, vmax / 2, vmax]) {
      el("line", {x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: "#2e3432"}, svg);
      el("text", {x: L - 4, y: Y(t) + 3, "text-anchor": "end"}, svg).textContent = "$" + t.toFixed(2);
    }
    for (let p = 0; p <= 70; p += 10) el("text", {x: X(p), y: H - 6, "text-anchor": "middle"}, svg).textContent = p + "%";
    el("polyline", {points: c.map(p => `${X(p.pct)},${Y(p.value)}`).join(" "), fill: "none", stroke: "var(--series-1)",
      "stroke-width": 2, "stroke-linejoin": "round"}, svg);
    if (o.action === "donate") {
      el("line", {x1: L, x2: W - R, y1: Y(o.expected_value), y2: Y(o.expected_value), stroke: "#c3c2b7", "stroke-dasharray": "4 3"}, svg);
      el("text", {x: W - R, y: Y(o.expected_value) - 4, "text-anchor": "end", class: "zone"}, svg).textContent = "donate value";
    } else {
      el("circle", {cx: X(o.markdown_pct), cy: Y(o.expected_value), r: 5, fill: "var(--series-1)", stroke: "#222625", "stroke-width": 2}, svg);
    }
    tipLayer(box, svg, W, L, W - R, sx => {
      const pv = (sx - L) / (W - L - R) * 70;
      const p = c.reduce((a, b) => Math.abs(b.pct - pv) < Math.abs(a.pct - pv) ? b : a);
      return {x: X(p.pct), yb: H - B, html: `<b>${p.pct}% off</b> · $${(base * (1 - p.pct / 100)).toFixed(2)}<br>` +
        `sells before spoiling ${pct(p.p_sell)} · expected $${p.value.toFixed(2)}`};
    });
  }

  // ------------------------------------------------------------------ cards
  let current = null, scenario = "counter";
  async function getJSON(u) { return (await fetch(u)).json(); }

  function render(root, r) {
    if (!r) { root.innerHTML = ""; current = null; return; }
    const same = current && current.scan_id === r.scan_id;
    if (same) { updateLate(root, r); current = r; return; }
    current = r; scenario = "counter";
    const cards = [];
    // Why
    cards.push(`<div class="card" id="c-why"><h3>Why this verdict</h3>
      <div class="pair"><figure><img src="${esc(r.image || "")}" alt="scan"><figcaption>what the camera saw</figcaption></figure>
      <figure id="why-fig"><div class="muted" style="padding:30px 0;text-align:center">computing heatmap…</div></figure></div>
      <div id="why-txt" class="muted" style="margin-top:6px"></div>
      <div style="margin-top:6px">${(r.basis || []).map(b => `<div>• ${esc(b)}</div>`).join("")}</div></div>`);
    // Forecast
    if (r.forecast) {
      cards.push(`<div class="card" id="c-fc"><h3>Ripening forecast</h3>
        <div id="fc-head"></div><div class="chart" id="fc-chart"></div>
        <div class="chips" id="fc-chips"></div><div class="muted" id="fc-notes"></div></div>`);
      cards.push(`<div class="card"><h3>Your ${esc(r.fruit_type)} in <span id="fu-d">3</span> days <span class="flag">simulated</span></h3>
        <div class="pair"><figure><img src="${esc(r.image)}" alt="now"><figcaption>now</figcaption></figure>
        <figure><img id="fu-img" alt="future"><figcaption id="fu-cap">in 3 days</figcaption></figure></div>
        <input type="range" id="fu-range" min="0" max="7" step="0.5" value="3" aria-label="days ahead">
        <div class="muted">Colour and spots moved along the forecast ripeness; not a photo.</div></div>`);
    }
    // Profile
    const p = r.profile;
    if (p) {
      const n = p.nutrition || {};
      const tile = (v, u, k) => v == null ? "" : `<div><b>${typeof v === "number" && v >= 20 ? Math.round(v) : v}${u}</b><span>${k}</span></div>`;
      cards.push(`<div class="card"><h3>This ${esc(p.fruit)} right now <span class="flag">estimates</span></h3>
        <div class="kv">
          ${p.weight_g ? `<span>Weight</span><b>${p.weight_g} g <span class="muted">(${p.weight_source === "depth" ? "from 3D volume" : "typical size"})</span></b>` : ""}
          ${p.size_cm ? `<span>Size</span><b>${p.size_cm[0]} × ${p.size_cm[1]} cm</b>` : ""}
          ${p.brix != null ? `<span>Sweetness</span><b>${p.brix} °Brix</b>` : ""}
          ${p.starch_g_100g != null ? `<span>Starch</span><b>${p.starch_g_100g} g / 100 g</b>` : ""}
        </div>
        ${p.note ? `<div style="margin-top:6px;color:var(--text-primary)">${esc(p.note)}</div>` : ""}
        ${n.kcal != null ? `<div class="nut">${tile(n.kcal, "", "kcal")}${tile(n.sugar_g, " g", "sugar")}${tile(n.fibre_g, " g", "fibre")}${tile(n.vitamin_c_mg == null ? null : Math.round(n.vitamin_c_mg), "", "mg vit C")}${tile(n.potassium_mg == null ? null : Math.round(n.potassium_mg), "", "mg potassium")}</div>` : ""}
        <div style="margin-top:8px"><b style="color:var(--text-primary)">Best uses now</b><ul>${(p.uses || []).map(u => `<li>${esc(u)}</li>`).join("")}</ul></div>
        ${p.storage ? `<div style="margin-top:6px"><b style="color:var(--text-primary)">Storage</b> · ${esc(p.storage)}</div>` : ""}
        ${(p.ethylene.tips || []).length ? `<div style="margin-top:6px"><b style="color:var(--text-primary)">Ethylene</b><ul>${p.ethylene.tips.map(t => `<li>${esc(t)}</li>`).join("")}</ul></div>` : ""}
        <div style="margin-top:8px"><button class="act" id="rc-btn">Use it tonight → recipe</button><div id="rc"></div></div></div>`);
    }
    // Store
    const st = r.store || {}, o = st.optimiser;
    if (o) {
      cards.push(`<div class="card"><h3>Store: optimal markdown</h3>
        <div class="big">${o.action === "donate" ? "Donate today" : (o.markdown_pct ? `$${o.price.toFixed(2)} <span class="muted">(${o.markdown_pct}% off $${st.base_price.toFixed(2)})</span>` : `$${o.price.toFixed(2)} full price`)}</div>
        <div class="muted">${esc(st.store_action)} · sells before spoiling: ${pct(o.p_sell)} · vs fixed tier ${o.vs_tier >= 0 ? "+" : ""}$${o.vs_tier.toFixed(3)}/item</div>
        <div class="chart" id="pr-chart" style="margin-top:6px"></div>
        <div class="muted">Expected $ back per item at each discount (demand model in config.yaml). Waste avoided vs no markdown: ${(o.waste_avoided_kg * 1000).toFixed(0)} g, ${(o.co2e_avoided_kg * 1000).toFixed(0)} g CO₂e.</div></div>`);
    }
    // IR
    if (r.ir && r.ir.image) {
      const nb = r.ir.ir_bruise_count || 0;
      cards.push(`<div class="card"><h3>Hidden bruises (850 nm IR)</h3><img src="${esc(r.ir.image)}" alt="IR view">
        <div class="big" style="margin-top:6px">${nb ? `${nb} bruise${nb > 1 ? "s" : ""}` : "None found"}</div>
        <div class="muted">${nb ? `${(r.ir.ir_bruise_area_cm2 || 0).toFixed(1)} cm² · ` : ""}Bruised cells absorb near-infrared, so they show here hours before the skin browns.</div></div>`);
    }
    root.innerHTML = `<div class="ins">${cards.join("")}</div>`;
    if (r.forecast) setupForecast(root, r);
    if (o) priceChart(root.querySelector("#pr-chart"), o, st.base_price);
    const rb = root.querySelector("#rc-btn");
    if (rb) rb.onclick = async () => {
      rb.disabled = true; rb.textContent = "…";
      const x = await getJSON(`/api/recipe/${r.scan_id}`);
      rb.style.display = "none";
      root.querySelector("#rc").innerHTML = `<div style="margin-top:4px"><b style="color:var(--text-primary)">${esc(x.title)}</b> <span class="muted">${x.minutes ? x.minutes + " min · " : ""}${x.source}</span>
        <div class="muted">${(x.ingredients || []).map(esc).join(" · ")}</div><ol style="margin:4px 0 0 18px;padding:0">${(x.steps || []).map(s => `<li>${esc(s)}</li>`).join("")}</ol></div>`;
    };
    updateLate(root, r);
  }

  function updateLate(root, r) {
    const w = r.why, fig = root.querySelector("#why-fig");
    if (w && fig && !fig.dataset.done) {
      fig.dataset.done = 1;
      fig.innerHTML = `<img src="${esc(w.image)}" alt="heatmap"><figcaption>where the ${esc(w.model)} looked</figcaption>`;
      root.querySelector("#why-txt").textContent =
        `${w.model} → "${w.class}" ${pct(w.prob)}${w.where ? `; strongest evidence ${w.where}` : ""}`;
    } else if (!w && fig && r.fruit_type === "unknown") {
      fig.innerHTML = `<div class="muted" style="padding:30px 0;text-align:center">no fruit recognised</div>`;
    }
  }

  function setupForecast(root, r) {
    const head = root.querySelector("#fc-head"), chips = root.querySelector("#fc-chips");
    function show(f, wi) {
      const pw = f.peak_window, dl = f.days_left;
      head.innerHTML = `<div class="big">${f.q_now >= 1.75 ? "Past its peak" : pw.start_days <= 0.05 ? "At its best now" : "Best " + when(pw.start)}</div>
        <div class="muted">${pw.end_days > 0.05 ? "good until " + when(pw.end) + " · " : ""}spoils in ${dl[1].toFixed(1)} days (likely ${dl[0].toFixed(1)}–${dl[2].toFixed(1)}) · ${esc(f.scenario_label)}, ${f.temp_c} °C · ${f.n_scans} scan${f.n_scans > 1 ? "s" : ""}</div>`;
      forecastChart(root.querySelector("#fc-chart"), f);
      root.querySelector("#fc-notes").innerHTML = (f.notes || []).map(esc).join("<br>");
      if (wi) chips.innerHTML = Object.entries(wi).map(([k, v]) =>
        `<button data-k="${k}" class="${k === scenario ? "on" : ""}">${esc(v.label)}${k !== "counter" ? ` · ${v.delta_days > 0 ? "+" : ""}${v.delta_days} d` : ""}</button>`).join("");
      chips.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.k === scenario));
    }
    show(r.forecast, r.whatif);
    chips.addEventListener("click", async ev => {
      const k = ev.target.dataset && ev.target.dataset.k; if (!k) return;
      scenario = k;
      const x = await getJSON(`/api/forecast/${r.scan_id}?scenario=${k}`);
      show(x.forecast, null); future();
    });
    const rg = root.querySelector("#fu-range");
    function future() {
      const d = rg.value;
      root.querySelector("#fu-d").textContent = d;
      root.querySelector("#fu-cap").textContent = `in ${d} days (${scenario})`;
      root.querySelector("#fu-img").src = `/api/future/${r.scan_id}.jpg?days=${d}&scenario=${scenario}`;
    }
    let t; rg.addEventListener("input", () => { clearTimeout(t); t = setTimeout(future, 120); });
    future();
  }

  return {render, forecastChart, priceChart, when};
})();
