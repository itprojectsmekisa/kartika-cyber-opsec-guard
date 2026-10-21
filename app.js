import { decodeYolo, matchText, assessRisk, explainFallback } from './core.js';

const SIZE = 640; // model input
const MAX_SIDE = 1600; // keeps OCR and canvas memory sane on phones
const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };

const classes = await fetch('model/classes.json').then(r => r.json());
const sessionPromise = loadModel();
sessionPromise.then(() => { $('progress').hidden = true; status('Siap. Pilih foto atau coba contoh.'); },
  (e) => status(`Model gagal dimuat: ${e.message}. Muat ulang halaman untuk mencoba lagi.`));

// shortcut: whole model (~50 MB) downloads on first visit, fine for demo on Wi-Fi; switch to a trained YOLO11n (~10 MB) after week 2.
async function loadModel() {
  const index = await fetch('model/parts.json');
  if (!index.ok) throw new Error('file model belum tersedia (jalankan scripts/export_model.py)');
  const count = await index.json();
  const loaded = [], total = [];
  const report = () => {
    const done = loaded.reduce((a, b) => a + b, 0), all = total.reduce((a, b) => a + b, 0);
    if (!all) return;
    $('progress').value = done / all;
    status(`Memuat model AI… ${Math.round((done / all) * 100)}% (sekali saja, lalu tersimpan di browser)`);
  };
  const parts = await Promise.all([...Array(count).keys()].map(async (i) => {
    const res = await fetch(`model/model.onnx.part${i}`);
    if (!res.ok) throw new Error(`bagian model ${i} tidak ditemukan`);
    total[i] = Number(res.headers.get('Content-Length')) || 0;
    loaded[i] = 0;
    const chunks = [];
    for (const reader = res.body.getReader(); ;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded[i] += value.length;
      report();
    }
    return chunks;
  }));
  const flat = parts.flat();
  const model = new Uint8Array(flat.reduce((n, c) => n + c.length, 0));
  flat.reduce((offset, c) => (model.set(c, offset), offset + c.length), 0);
  status('Menyiapkan model AI…');
  return ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
}

async function run(file) {
  if (document.body.classList.contains('busy')) return;
  document.body.classList.add('busy');
  document.querySelectorAll('.sample').forEach(b => { b.disabled = true; });
  try {
    await analyze(file);
  } catch (err) {
    status(`Gagal memeriksa foto: ${err.message}`);
  } finally {
    document.body.classList.remove('busy');
    document.querySelectorAll('.sample').forEach(b => { b.disabled = false; });
    $('file').value = ''; // lets the same photo be picked again
  }
}

$('file').addEventListener('change', (e) => e.target.files[0] && run(e.target.files[0]));

for (const btn of document.querySelectorAll('.sample')) {
  btn.addEventListener('click', async () => {
    const blob = await fetch(btn.dataset.src).then(r => r.blob());
    run(new File([blob], btn.dataset.src.split('/').pop(), { type: blob.type }));
  });
}

const drop = $('drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) run(file);
});

async function analyze(file) {
  status('Membaca foto…');
  const bitmap = await createImageBitmap(file);
  const k = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const base = canvas(Math.round(bitmap.width * k), Math.round(bitmap.height * k));
  base.getContext('2d').drawImage(bitmap, 0, 0, base.width, base.height);

  status('Menganalisis: deteksi objek, membaca tulisan, memeriksa lokasi GPS…');
  const [detections, words, gps] = await Promise.all([
    detect(base),
    Tesseract.recognize(base, 'ind+eng').then(r => r.data.words),
    exifr.gps(file).catch(() => null),
  ]);

  const result = assessRisk({ detections, classes, textHits: matchText(words), gps });
  render(base, result);
  status(`Selesai: ${result.findings.length} temuan.`);
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('verdict-title').focus({ preventScroll: true });
  $('explain-box').classList.add('loading');
  $('explain').textContent = 'Menyusun penjelasan…';
  $('explain').textContent = await explain(result);
  $('explain-box').classList.remove('loading');
}

async function detect(base) {
  const session = await sessionPromise;
  const scale = Math.min(SIZE / base.width, SIZE / base.height);
  const padX = (SIZE - base.width * scale) / 2, padY = (SIZE - base.height * scale) / 2;
  const box = canvas(SIZE, SIZE);
  const ctx = box.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(base, padX, padY, base.width * scale, base.height * scale);

  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const area = SIZE * SIZE, input = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    input[i] = px[i * 4] / 255;
    input[area + i] = px[i * 4 + 1] / 255;
    input[2 * area + i] = px[i * 4 + 2] / 255;
  }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
  const t = out[session.outputNames[0]];
  return decodeYolo(t.data, t.dims[2], t.dims[1] - 4, { scale, padX, padY });
}

function render(base, { level, findings }) {
  $('result').hidden = false;
  $('level').textContent = `RISIKO ${level}`;
  $('level').className = `badge ${level}`;
  const sorted = [...findings].sort((x, y) => y.severity - x.severity);
  $('findings').replaceChildren(...(sorted.length ? sorted : [{ label: 'Tidak ada temuan', detail: '' }]).map(f => {
    const li = document.createElement('li');
    li.dataset.sev = f.severity ?? 0;
    li.append(f.label, ' ');
    li.appendChild(document.createElement('small')).textContent = f.detail;
    return li;
  }));

  const annotated = copyInto($('annotated'), base);
  const a = annotated.getContext('2d');
  a.lineWidth = Math.max(2, base.width / 300);
  const size = Math.max(14, base.width / 50);
  a.font = `600 ${size}px system-ui`;
  for (const f of findings.filter(f => f.box)) {
    const color = f.severity >= 3 ? '#e53935' : f.severity === 2 ? '#fb8c00' : '#fdd835';
    a.strokeStyle = color;
    a.strokeRect(f.box.x, f.box.y, f.box.w, f.box.h);
    // Label on a solid tag so it stays readable on busy photos.
    const pad = size * 0.3, w = a.measureText(f.label).width + pad * 2, h = size + pad * 2;
    const y = f.box.y >= h ? f.box.y - h : f.box.y;
    a.fillStyle = color;
    a.fillRect(f.box.x, y, w, h);
    a.fillStyle = f.severity === 1 ? '#000' : '#fff';
    a.fillText(f.label, f.box.x + pad, y + pad + size * 0.85);
  }

  // Re-encoding through canvas drops all EXIF metadata, including GPS.
  const safe = copyInto($('safe'), base);
  for (const f of findings.filter(f => f.box && f.blur)) pixelate(safe, f.box);
  safe.toBlob((blob) => {
    URL.revokeObjectURL($('download').href);
    $('download').href = URL.createObjectURL(blob);
  }, 'image/jpeg', 0.92);
}

async function explain(result) {
  try {
    const res = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level: result.level, findings: result.labels }),
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) throw new Error(res.status);
    return (await res.json()).text || explainFallback(result);
  } catch {
    return explainFallback(result);
  }
}

function canvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function copyInto(target, base) {
  target.width = base.width; target.height = base.height;
  target.getContext('2d').drawImage(base, 0, 0);
  return target;
}

// Pixelate instead of ctx.filter blur: works on every mobile browser.
function pixelate(target, { x, y, w, h }) {
  [x, y] = [Math.max(0, x), Math.max(0, y)];
  [w, h] = [Math.min(w, target.width - x), Math.min(h, target.height - y)];
  if (w < 1 || h < 1) return;
  const small = canvas(Math.max(1, Math.round(w / 16)), Math.max(1, Math.round(h / 16)));
  small.getContext('2d').drawImage(target, x, y, w, h, 0, 0, small.width, small.height);
  const ctx = target.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, small.width, small.height, x, y, w, h);
  ctx.imageSmoothingEnabled = true;
}
