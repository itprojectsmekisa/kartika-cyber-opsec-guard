# Kartika Cyber OPSEC Guard

Asisten AI pencegah kebocoran informasi prajurit di media sosial. Foto diperiksa **di perangkat pengguna**:

- **Deteksi objek** (YOLO-World, ONNX di browser): senjata, kendaraan militer, dokumen, peta, layar, pelat nomor, papan nama, wajah.
- **OCR** (Tesseract.js): tulisan "RAHASIA", nama satuan (YONIF, KODIM, …), pelat nomor militer.
- **EXIF**: lokasi GPS yang tersimpan di file foto.
- **Versi aman**: bagian sensitif dipikselkan, metadata GPS dihapus.
- **Penjelasan AI** (DeepSeek V4 Flash lewat EdgeOne Makers AI Gateway): hanya menerima daftar jenis temuan, tanpa foto dan koordinat. Kalau tidak tersedia, aplikasi memakai penjelasan bawaan.

## Struktur

| Path | Isi |
|---|---|
| `index.html`, `app.js` | Aplikasi web (tanpa build step) |
| `core.js`, `core.test.js` | Logika murni: dekode YOLO, pencocokan teks, penilaian risiko, dan tesnya |
| `model/classes.json` | Daftar kelas deteksi, label, dan bobot risiko (sumber tunggal untuk export, app, dan function) |
| `model/model.onnx.part*` | Model hasil export, dipecah karena batas 25 MB per file di EdgeOne |
| `edge-functions/api/explain/` | Endpoint `/api/explain` yang memanggil DeepSeek |
| `scripts/export_model.py` | Export YOLO-World + prompt kelas ke ONNX |

## Menjalankan

```bash
npm test                                   # tes logika inti
python3 -m venv .venv && .venv/bin/pip install ultralytics onnx onnxslim
.venv/bin/python scripts/export_model.py   # membuat model/model.onnx.part*
python3 -m http.server 8791                # buka http://localhost:8791 (tanpa penjelasan AI)
```

## Deploy ke Tencent EdgeOne

```bash
npm install -g edgeone
edgeone login                              # pilih Global, selesaikan di browser
edgeone makers dev                         # uji lokal lengkap dengan edge function di http://localhost:8088
npm run deploy                             # build ke dist/ lalu deploy hanya file publik
```
