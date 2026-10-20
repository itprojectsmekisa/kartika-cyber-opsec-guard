import { decodeYolo, matchText, assessRisk, explainFallback } from './core.js';

const SIZE = 640; // model input
const MAX_SIDE = 1600; // keeps OCR and canvas memory sane on phones
const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };

const classes = await fetch('model/classes.json').then(r => r.json());
const sessionPromise = loadModel();
sessionPromise.then(() => status('Siap. Pilih foto untuk diperiksa.'), (e) => status(`Model gagal dimuat: ${e.message}`));

// shortcut: whole model (~50 MB) downloads on first visit, fine for demo on Wi-Fi; switch to a trained YOLO11n (~10 MB) after week 2.
async function loadModel() {
  const index = await fetch('model/parts.json');
  if (!index.ok) throw new Error('file model belum tersedia (jalankan scripts/export_model.py)');
  const count = await index.json();
  const parts = await Promise.all([...Array(count).keys()].map(async (i) => {
    const res = await fetch(`model/model.onnx.part${i}`);
    if (!res.ok) throw new Error(`bagian model ${i} tidak ditemukan`);
    return new Uint8Array(await res.arrayBuffer());
  }));
  const model = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((offset, p) => (model.set(p, offset), offset + p.length), 0);
  return ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
}

$('file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    await analyze(file);
  } catch (err) {
    status(`Gagal memeriksa foto: ${err.message}`);
  }
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
  status('Selesai.');
  $('explain').textContent = 'Menyusun penjelasan…';
  $('explain').textContent = await explain(result);
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
  $('findings').replaceChildren(...(findings.length ? findings : [{ label: 'Tidak ada temuan', detail: '' }]).map(f => {
    const li = document.createElement('li');
    li.append(f.label, ' ');
    li.appendChild(document.createElement('small')).textContent = f.detail;
    return li;
  }));

  const annotated = copyInto($('annotated'), base);
  const a = annotated.getContext('2d');
  a.lineWidth = Math.max(2, base.width / 300);
  a.font = `${Math.max(14, base.width / 50)}px system-ui`;
  for (const f of findings.filter(f => f.box)) {
    a.strokeStyle = a.fillStyle = f.severity >= 3 ? '#e53935' : f.severity === 2 ? '#fb8c00' : '#fdd835';
    a.strokeRect(f.box.x, f.box.y, f.box.w, f.box.h);
    a.fillText(f.label, f.box.x, Math.max(16, f.box.y - 4));
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
    const text = await readSse(res);
    return text || explainFallback(result);
  } catch {
    return explainFallback(result);
  }
}

// OpenAI-style SSE: lines of `data: {"choices":[{"delta":{"content":"..."}}]}`.
async function readSse(res) {
  const raw = await res.text();
  let text = '';
  for (const line of raw.split('\n')) {
    if (!line.startsWith('data:') || line.includes('[DONE]')) continue;
    try { text += JSON.parse(line.slice(5)).choices?.[0]?.delta?.content ?? ''; } catch { /* partial keep-alive line */ }
  }
  return text.trim();
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
