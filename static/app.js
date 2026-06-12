"use strict";

let SPEC = null;
let METRICS = null;

const $ = (id) => document.getElementById(id);
const fmtMoney = (v) => "NT$" + Math.round(v).toLocaleString();
const isMoney = (name) => name.includes("AMT") || name === "LIMIT_BAL";

async function init() {
  [SPEC, METRICS] = await Promise.all([
    fetch("/spec").then((r) => r.json()),
    fetch("/metrics").then((r) => r.json()),
  ]);

  $("auc-badge").textContent = "AUC " + METRICS.auc_roc;
  buildInputs();

  $("threshold").addEventListener("input", () => {
    $("threshold-val").textContent = Number($("threshold").value).toFixed(2);
    score();
  });

  score();
}

function buildInputs() {
  const host = $("inputs");
  host.innerHTML = "";
  for (const f of SPEC.features) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const valId = `v_${f.name}`;
    wrap.innerHTML = `
      <div class="row-between">
        <label for="i_${f.name}">${f.label}</label>
        <span class="val" id="${valId}"></span>
      </div>
      <input type="range" id="i_${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}" />
      ${f.help ? `<div class="hint">${f.help}</div>` : ""}`;
    host.appendChild(wrap);

    const input = $(`i_${f.name}`);
    const show = () => {
      $(valId).textContent = isMoney(f.name) ? fmtMoney(input.value) : input.value;
    };
    show();
    input.addEventListener("input", () => { show(); score(); });
  }
}

function collect() {
  const features = {};
  for (const f of SPEC.features) features[f.name] = Number($(`i_${f.name}`).value);
  return features;
}

let pending = null;
function score() {
  // light debounce so dragging a slider doesn't flood the API
  clearTimeout(pending);
  pending = setTimeout(doScore, 120);
}

async function doScore() {
  const threshold = Number($("threshold").value);
  const res = await fetch("/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features: collect(), threshold }),
  });
  if (!res.ok) {
    $("prob").textContent = "error";
    return;
  }
  render(await res.json(), threshold);
}

function render(r, threshold) {
  const pct = r.probability * 100;
  $("prob").textContent = pct.toFixed(1) + "%";

  const dec = $("decision");
  dec.textContent = r.decision === "approve" ? "✓ Approve" : "✕ Decline";
  dec.className = "decision " + r.decision;

  $("meter-fill").style.width = (100 - pct) + "%";
  $("meter-mark").style.left = (threshold * 100) + "%";

  renderBusiness(threshold);
  renderShap(r.contributions);
}

function renderBusiness(threshold) {
  const curve = METRICS.threshold_curve || [];
  if (!curve.length) { $("business").innerHTML = ""; return; }
  // nearest threshold entry
  let best = curve[0];
  for (const row of curve) {
    if (Math.abs(row.threshold - threshold) < Math.abs(best.threshold - threshold)) best = row;
  }
  const cell = (v, k) => `<div class="stat"><div class="v">${(v * 100).toFixed(0)}%</div><div class="k">${k}</div></div>`;
  $("business").innerHTML =
    cell(best.approval_rate, "approved (test set)") +
    cell(best.recall, "defaulters caught") +
    cell(best.precision, "of declines truly default");
}

function renderShap(contribs) {
  const host = $("shap");
  host.innerHTML = "";
  const maxAbs = Math.max(...contribs.map((c) => Math.abs(c.contribution)), 1e-6);
  for (const c of contribs) {
    const frac = (Math.abs(c.contribution) / maxAbs) * 50; // % of half-track
    const pos = c.contribution >= 0;
    const valLabel = isMoney(c.feature) ? fmtMoney(c.value) : c.value;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="name" title="${c.feature} = ${valLabel}">${c.feature} <span class="muted">= ${valLabel}</span></div>
      <div class="bar-track">
        <div class="bar-mid"></div>
        <div class="bar ${pos ? "pos" : "neg"}" style="width:${frac}%"></div>
      </div>`;
    host.appendChild(row);
  }
}

init();
