// Pure logic shared by the browser app and the tests: no DOM here.

const SECRET_WORDS = ['RAHASIA', 'TERBATAS', 'KONFIDENSIAL', 'SECRET', 'CONFIDENTIAL'];
const UNIT_WORDS = ['YONIF', 'KODIM', 'KOREM', 'KODAM', 'BATALYON', 'KOSTRAD', 'KOPASSUS', 'DENMA', 'RINDAM',
  'YONKAV', 'YONARMED', 'YONZIPUR', 'PUSSIBERAD', 'MAKO', 'MARKAS', 'LANUD', 'KOMPI', 'PELETON'];
const PLATE = /^\d{3,5}-\d{2}$/; // TNI vehicle plates look like 12345-00
const LEVELS = ['AMAN', 'RENDAH', 'SEDANG', 'TINGGI'];

// YOLO output laid out as [1, 4 + numClasses, numBoxes] with cx,cy,w,h in letterboxed model pixels.
export function decodeYolo(data, numBoxes, numClasses, { scale, padX, padY }, minScore = 0.25, maxIou = 0.45) {
  const found = [];
  for (let i = 0; i < numBoxes; i++) {
    let cls = 0, score = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * numBoxes + i];
      if (s > score) { score = s; cls = c; }
    }
    if (score < minScore) continue;
    const [cx, cy, w, h] = [0, 1, 2, 3].map(k => data[k * numBoxes + i]);
    found.push({ cls, score, x: (cx - w / 2 - padX) / scale, y: (cy - h / 2 - padY) / scale, w: w / scale, h: h / scale });
  }
  found.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of found) {
    if (!kept.some(k => k.cls === box.cls && iou(k, box) > maxIou)) kept.push(box);
  }
  return kept;
}

function iou(a, b) {
  const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  const inter = Math.max(0, w) * Math.max(0, h);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// words: Tesseract.js words, each { text, bbox: { x0, y0, x1, y1 } }.
export function matchText(words) {
  const hits = [];
  for (const { text, bbox } of words) {
    const t = text.toUpperCase().replace(/[^A-Z0-9-]/g, '');
    const kind = SECRET_WORDS.includes(t) ? 'secret' : UNIT_WORDS.includes(t) ? 'unit' : PLATE.test(t) ? 'plate' : null;
    if (kind) hits.push({ kind, text: t, box: { x: bbox.x0, y: bbox.y0, w: bbox.x1 - bbox.x0, h: bbox.y1 - bbox.y0 } });
  }
  return hits;
}

const TEXT_FINDINGS = {
  secret: { label: 'Tulisan klasifikasi rahasia terbaca', severity: 3 },
  unit: { label: 'Nama satuan terbaca', severity: 2 },
  plate: { label: 'Nomor kendaraan militer terbaca', severity: 2 },
};

// label is generic and may be sent to the server; detail stays in the browser.
export function assessRisk({ detections, classes, textHits, gps }) {
  const findings = [
    ...detections.map(d => ({ ...classes[d.cls], blur: classes[d.cls].blur !== false, box: d, detail: `${Math.round(d.score * 100)}% yakin` })),
    ...textHits.map(h => ({ ...TEXT_FINDINGS[h.kind], blur: true, box: h.box, detail: `"${h.text}"` })),
  ];
  if (gps) findings.push({ label: 'Lokasi GPS tersimpan di file foto', severity: 3, blur: false, detail: `${gps.latitude.toFixed(5)}, ${gps.longitude.toFixed(5)}` });

  const max = Math.max(0, ...findings.map(f => f.severity));
  const mediumByCount = max === 1 && findings.length >= 2;
  const level = LEVELS[mediumByCount ? 2 : max];
  return { level, findings, labels: [...new Set(findings.map(f => f.label))] };
}

export function explainFallback({ level, labels }) {
  if (level === 'AMAN') return 'Tidak ditemukan informasi sensitif. Tetap periksa latar belakang foto sebelum mengunggah.';
  return `Risiko ${level}. Ditemukan: ${labels.join(', ')}. Informasi ini bisa dirangkai lawan untuk mengetahui lokasi, ` +
    'kekuatan, atau kegiatan satuan. Jangan unggah foto asli; gunakan versi aman di bawah dan matikan lokasi pada kamera HP.';
}
