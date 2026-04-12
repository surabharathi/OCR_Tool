/**
 * app.js — Form Extractor single-page UI (single-engine, single-phase).
 *
 * State machine:  idle → scanning → complete
 */
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let scanId     = null;
let evtSource  = null;
let totalImages = 0;
let doneImages  = 0;
let skipImages  = 0;
let failImages  = 0;
let scanStartTime = null;

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => $(id).classList.remove('hidden');
const hide = id => $(id).classList.add('hidden');
const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };

function setStep(name) {
  document.querySelectorAll('.step-item').forEach(el => el.classList.remove('active'));
  const t = $(`step-${name}`);
  if (t) t.classList.add('active');
}
function markStepDone(name) {
  const el = $(`step-${name}`);
  if (el) { el.classList.remove('active'); el.classList.add('done'); }
}

function confBadge(score) {
  const n = parseFloat(score) || 0;
  if (n >= 80) return `<span class="badge badge-green">${n.toFixed(0)}%</span>`;
  if (n >= 55) return `<span class="badge badge-amber">${n.toFixed(0)}%</span>`;
  return `<span class="badge badge-red">${n.toFixed(0)}%</span>`;
}

function truncate(str, n = 28) {
  if (!str) return '—';
  return str.length > n ? str.slice(0, n) + '…' : str;
}

function imgsPerMin() {
  if (!scanStartTime || doneImages === 0) return '—';
  const mins = (Date.now() - scanStartTime) / 60000;
  return mins > 0 ? Math.round(doneImages / mins) : '—';
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = $('health-dot');
  const text = $('health-text');
  dot.className = 'health-dot';
  text.textContent = 'Checking…';

  const model   = $('ollama-model').value.trim() || 'reducto/rolmocr';
  const baseUrl = $('ollama-url').value.trim() || 'http://localhost:11434';

  try {
    const res  = await fetch(`/api/health?model=${encodeURIComponent(model)}&base_url=${encodeURIComponent(baseUrl)}`);
    const data = await res.json();

    if (data.ollama_ok) {
      dot.classList.add('ok');
      text.textContent = `${model} · ready`;
      setText('model-display', `${model} · RTX 4090`);
    } else {
      dot.classList.add('err');
      text.textContent = `Model not found — run: ollama pull ${model}`;
    }
  } catch {
    dot.classList.add('err');
    text.textContent = 'Ollama unreachable';
  }
}

// ── Start scan ────────────────────────────────────────────────────────────────
async function startScan() {
  const imageFolder = $('image-folder').value.trim();
  if (!imageFolder) { alert('Please enter an image folder path.'); return; }

  const model    = $('ollama-model').value.trim() || 'reducto/rolmocr';
  const baseUrl  = $('ollama-url').value.trim()   || 'http://localhost:11434';
  const fresh    = document.querySelector('input[name="run-mode"]:checked').value === 'fresh';

  $('btn-start').disabled = true;
  $('btn-start').textContent = '⏳ Starting…';

  try {
    const res = await fetch('/api/scan/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_folder: imageFolder,
        model:        model,
        base_url:     baseUrl,
        fresh_start:  fresh,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Error: ${err.error || res.statusText}`);
      resetStartButton();
      return;
    }

    const data = await res.json();
    scanId = data.scan_id;
    scanStartTime = Date.now();

    hide('section-configure');
    show('section-scan');
    show('btn-stop');
    show('sidebar-stats');
    setText('scan-model-label', model);
    setStep('scan');
    setText('scan-status-text', `Processing with ${model} on local GPU…`);

    openSSEStream();
  } catch (err) {
    alert(`Network error: ${err.message}`);
    resetStartButton();
  }
}

function resetStartButton() {
  $('btn-start').disabled = false;
  $('btn-start').innerHTML = '<span class="btn-icon">▶</span> Start GPU Scan';
}

// ── SSE stream ────────────────────────────────────────────────────────────────
function openSSEStream() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource(`/api/scan/stream/${scanId}`);
  evtSource.onmessage = e => {
    try { handleEvent(JSON.parse(e.data)); }
    catch (ex) { console.error('SSE parse error:', ex, e.data); }
  };
  evtSource.onerror = () => console.warn('SSE connection lost — retrying…');
}

// ── Event handler ─────────────────────────────────────────────────────────────
function handleEvent(evt) {
  switch (evt.type) {
    case 'scan_start':
      totalImages = evt.total;
      updateCounters();
      break;

    case 'image_done':
      if (evt.status === 'processed') { doneImages++; addResultRow(evt); }
      else if (evt.status.startsWith('skipped')) skipImages++;
      else if (evt.status === 'failed') failImages++;
      updateCounters();
      updateProgress(evt.progress || 0);
      break;

    case 'scan_complete':
      evtSource.close();
      onScanComplete(evt);
      break;

    case 'stream_end':
      evtSource.close();
      break;

    case 'error':
      alert(`Scan error: ${evt.message}`);
      evtSource.close();
      resetStartButton();
      break;

    case 'heartbeat':
      setText('c-rate', imgsPerMin());
      break;
  }
}

// ── Counter updates ───────────────────────────────────────────────────────────
function updateCounters() {
  setText('c-done',    doneImages);
  setText('c-skipped', skipImages);
  setText('c-failed',  failImages);
  setText('c-rate',    imgsPerMin());
  setText('s-processed', doneImages);
  setText('s-skipped',   skipImages);
  setText('s-failed',    failImages);
}

function updateProgress(pct) {
  const processed = doneImages + skipImages + failImages;
  setText('progress-label', `${processed} / ${totalImages} images`);
  setText('progress-pct',   `${pct.toFixed(0)}%`);
  $('progress-bar').style.width = `${pct}%`;
}

// ── Live result row ───────────────────────────────────────────────────────────
function addResultRow(evt) {
  show('results-wrapper');
  const f  = evt.fields || {};
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td class="mono" title="${evt.filename}">${truncate(evt.filename, 22)}</td>
    <td class="mono">${f.app_no || '—'}</td>
    <td title="${f.name || ''}">${truncate(f.name, 20)}</td>
    <td title="${(f.address_line_1 || '') + ', ' + (f.address_line_2 || '') + ', ' + (f.address_line_3 || '')}">${truncate((f.address_line_1 || '') + ', ' + (f.address_line_2 || '') + ', ' + (f.address_line_3 || ''), 26)}</td>
    <td class="mono">${f.mob_no || '—'}</td>
    <td title="${f.email || ''}">${truncate(f.email, 20)}</td>
    <td>${f.q1_chanted_before || '—'}</td>
    <td>${confBadge(evt.avg_confidence)}</td>
    <td><span class="badge badge-green">Done</span></td>
  `;
  $('results-body').prepend(tr);
}

// ── Scan complete ─────────────────────────────────────────────────────────────
function onScanComplete(evt) {
  markStepDone('configure');
  markStepDone('scan');
  setStep('complete');
  hide('section-scan');
  hide('section-configure');
  hide('btn-stop');
  updateProgress(100);

  const stats = evt.stats || {};
  const processed = stats.processed ?? doneImages;
  const skipped = stats.skipped_duplicates ?? skipImages;
  const failed = stats.failed ?? failImages;
  const total = processed + skipped + failed;

  if (evt.csv_file) {
    setText('csv-path', evt.csv_file);
    show('csv-path-block');
  }

  $('final-stats').innerHTML = `
    <div class="counter green">
      <div class="counter-num">${processed}</div>
      <div class="counter-lbl">Processed</div>
    </div>
    <div class="counter blue">
      <div class="counter-num">${skipped}</div>
      <div class="counter-lbl">Skipped</div>
    </div>
    <div class="counter red">
      <div class="counter-num">${failed}</div>
      <div class="counter-lbl">Failed</div>
    </div>
    <div class="counter amber">
      <div class="counter-num">${imgsPerMin()}</div>
      <div class="counter-lbl">img / min</div>
    </div>
  `;

  setText('complete-summary', failed > 0 ? `${total} images processed, ${failed} failed.` : `${total} images processed. Zero API cost.`);
  setText('c-done', processed);
  setText('c-skipped', skipped);
  setText('c-failed', failed);
  setText('s-processed', processed);
  setText('s-skipped', skipped);
  setText('s-failed', failed);

  show('section-complete');
}

// ── Stop ──────────────────────────────────────────────────────────────────────
async function stopScan() {
  if (!scanId) return;
  try {
    await fetch(`/api/scan/stop/${scanId}`, { method: 'POST' });
    setText('scan-status-text', 'Stop requested — finishing current image…');
    $('btn-stop').disabled = true;
  } catch (err) { console.error('Stop request failed:', err); }
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetUI() {
  if (evtSource) evtSource.close();
  scanId = null;
  totalImages = doneImages = skipImages = failImages = 0;
  scanStartTime = null;

  ['section-scan', 'section-complete', 'btn-stop',
   'sidebar-stats', 'results-wrapper'].forEach(hide);
  show('section-configure');
  $('results-body').innerHTML = '';
  $('progress-bar').style.width = '0%';
  ['c-done','c-skipped','c-failed'].forEach(id => setText(id, '0'));
  setText('c-rate', '—');

  document.querySelectorAll('.step-item').forEach(el =>
    el.classList.remove('active','done')
  );
  $('step-configure').classList.add('active');
  resetStartButton();
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  // Auto-recheck health when model or URL fields change.
  ['ollama-model','ollama-url'].forEach(id => {
    const el = $(id);
    if (el) el.addEventListener('change', checkHealth);
  });
});
