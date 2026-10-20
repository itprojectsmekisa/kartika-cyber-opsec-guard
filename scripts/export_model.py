# Export YOLO-World with our OPSEC prompts to ONNX for onnxruntime-web, split into
# chunks because EdgeOne serves at most 25 MB per file.
# Run: .venv/bin/python scripts/export_model.py
import json
from pathlib import Path

from ultralytics import YOLOWorld

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
CHUNK = 20 * 1024 * 1024

prompts = [c["prompt"] for c in json.loads((MODEL_DIR / "classes.json").read_text())]
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(prompts)
onnx_path = Path(model.export(format="onnx", imgsz=640, simplify=True))

data = onnx_path.read_bytes()
for old in MODEL_DIR.glob("model.onnx.part*"):
    old.unlink()
parts = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
for i, part in enumerate(parts):
    (MODEL_DIR / f"model.onnx.part{i}").write_bytes(part)
(MODEL_DIR / "parts.json").write_text(json.dumps(len(parts)))
print(f"{len(data) / 1e6:.1f} MB -> {len(parts)} parts in {MODEL_DIR}")
