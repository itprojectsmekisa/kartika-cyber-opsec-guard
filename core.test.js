import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeYolo, matchText, assessRisk, explainFallback } from './core.js';

test('decodeYolo maps letterboxed boxes back and removes overlaps of the same class', () => {
  // 3 boxes, 2 classes, layout [cx, cy, w, h, score0, score1] x boxes
  const boxes = [
    [100, 100, 40, 40, 0.9, 0.1], // class 0, kept
    [102, 101, 40, 40, 0.8, 0.0], // overlaps box 0, same class -> dropped
    [300, 300, 20, 20, 0.1, 0.1], // below threshold -> dropped
  ];
  const n = boxes.length;
  const data = new Float32Array(6 * n);
  boxes.forEach((b, i) => b.forEach((v, k) => { data[k * n + i] = v; }));

  const out = decodeYolo(data, n, 2, { scale: 0.5, padX: 0, padY: 20 });
  assert.equal(out.length, 1);
  assert.equal(out[0].cls, 0);
  assert.deepEqual([out[0].x, out[0].y, out[0].w, out[0].h], [160, 120, 80, 80]);
});

test('matchText finds secret words, unit names and military plates only', () => {
  const box = { x0: 0, y0: 0, x1: 10, y1: 5 };
  const hits = matchText([
    { text: 'Rahasia,', bbox: box }, { text: 'YONIF', bbox: box }, { text: '12345-00', bbox: box }, { text: 'halo', bbox: box },
  ]);
  assert.deepEqual(hits.map(h => h.kind), ['secret', 'unit', 'plate']);
});

test('assessRisk levels and GPS coordinates stay out of the shareable labels', () => {
  const classes = [{ label: 'Senjata', severity: 3 }, { label: 'Wajah', severity: 1 }, { label: 'Seragam', severity: 1, blur: false }];
  const det = (cls) => ({ cls, score: 0.9, x: 0, y: 0, w: 1, h: 1 });

  assert.equal(assessRisk({ detections: [], classes, textHits: [], gps: null }).level, 'AMAN');
  assert.equal(assessRisk({ detections: [det(1)], classes, textHits: [], gps: null }).level, 'RENDAH');
  assert.equal(assessRisk({ detections: [det(1), det(2)], classes, textHits: [], gps: null }).level, 'SEDANG');

  const r = assessRisk({ detections: [det(0), det(2)], classes, textHits: [], gps: { latitude: -6.2, longitude: 106.8 } });
  assert.equal(r.level, 'TINGGI');
  assert.ok(!JSON.stringify(r.labels).includes('106.8'));
  assert.equal(r.findings.find(f => f.label === 'Seragam').blur, false);
  assert.match(explainFallback(r), /^Ditemukan: Senjata/);
});
