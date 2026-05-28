import sys
import io
import base64
import random
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
sys.path.insert(0, str(ROOT))

import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from services.model_factory import ModelFactory
from utils.config import Config
from utils.transforms import SolarTransforms
from config.GetPath import paths

_EVAL_TRANSFORM = SolarTransforms(img_size = Config.IMG_SIZE).eval

app = Flask(__name__, static_folder = str(FRONTEND), static_url_path = "")
CORS(app)

class DataService:

    CLASS_NAMES = Config.CLASS_NAMES # ["Clean","Dusty","Snow"]
    CLASS_DIRS  = {
        "Clean": paths.data_clean,
        "Dusty": paths.data_dusty,
        "Snow": paths.data_snow
    }

    @classmethod
    def get_class_counts(cls) -> dict:
        counts = {}
        for name, d in cls.CLASS_DIRS.items():
            counts[name] = len(list(d.iterdir())) if d.exists() else 0
        return counts

    @classmethod
    def get_distribution_stats(cls) -> dict:
        counts = cls.get_class_counts()
        total = sum(counts.values())
        return {
            "counts": counts,
            "total": total,
            "classes": cls.CLASS_NAMES
        }

    @classmethod
    def get_sample_images(cls, class_name: str, n: int = 5) -> list[str]:
        d = cls.CLASS_DIRS.get(class_name)
        if not d or not d.exists():
            return []
        files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        chosen = random.sample(files, min(n, len(files)))
        result = []
        for f in chosen:
            try:
                img = Image.open(f).convert("RGB").resize((112, 112))
                buf = io.BytesIO()
                img.save(buf, format = "JPEG", quality = 75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                result.append(f"data:image/jpeg;base64, {b64}")
            except Exception:
                pass
        return result

class ModelService:
   
    KEY_TO_NAME = {
        "efficientnetb4": "EfficientNet-B4",
        "resnet50": "ResNet-50",
        "vit": "ViT-Base"
    }
    NAME_TO_KEY = {v: k for k, v in KEY_TO_NAME.items()}
    MODEL_ORDER = ["efficientnetb4", "resnet50", "vit"]

    _models: dict = {}
    _eval_cache: dict = {}
    _history_cache: dict = {}

    @classmethod
    def _load_json(cls, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[ModelService] Failed to read {path}: {e}")
            return None

    @classmethod
    def _get_eval(cls, key: str) -> dict | None:
        if key not in cls._eval_cache:
            cls._eval_cache[key] = cls._load_json(paths.models / f"{key}_eval.json")
        return cls._eval_cache[key]

    @classmethod
    def _get_history(cls, key: str) -> dict | None:
        if key not in cls._history_cache:
            cls._history_cache[key] = cls._load_json(paths.models / f"{key}_history.json")
        return cls._history_cache[key]

    @classmethod
    def get_metrics(cls) -> list:
        rows = []
        for key in cls.MODEL_ORDER:
            ev = cls._get_eval(key)
            if ev is None:
                continue
            rows.append({
                "name": cls.KEY_TO_NAME[key],
                "key": key,
                "acc": ev["accuracy"],
                "prec": ev["precision"],
                "rec": ev["recall"],
                "f1": ev["f1"],
                "auc": ev.get("auc"),
                "ms": ev.get("inference_ms"),
                "n_samples": ev.get("n_samples"),
                "per_class_f1": ev.get("per_class_f1"),
                "per_class_precision": ev.get("per_class_precision"),
                "per_class_recall": ev.get("per_class_recall")
            })
        if rows:
            best_idx = max(range(len(rows)), key=lambda i: rows[i]["f1"])
            for i, r in enumerate(rows):
                r["best"] = (i == best_idx)
        return rows

    @classmethod
    def get_confusion(cls, model_name: str) -> dict | None:
        key = cls.NAME_TO_KEY.get(model_name)
        if key is None:
            return None
        ev = cls._get_eval(key)
        if ev is None or "confusion_matrix" not in ev:
            return None
        return {
            "matrix": ev["confusion_matrix"],
            "class_names": ev.get("class_names", Config.CLASS_NAMES),
            "n_samples": ev.get("n_samples")
        }

    @classmethod
    def get_errors(cls, model_name: str, limit: int = 24) -> dict | None:
        key = cls.NAME_TO_KEY.get(model_name)
        if key is None:
            return None
        ev = cls._get_eval(key)
        if ev is None:
            return None
        errs = ev.get("errors", [])
        errs_sorted = sorted(errs, key = lambda e: e.get("confidence", 0), reverse = True)
        return {
            "errors": errs_sorted[:limit],
            "total_errors": len(errs),
            "n_samples": ev.get("n_samples")    
        }

    @classmethod
    def get_history(cls) -> dict:
        out = {}
        for key in cls.MODEL_ORDER:
            h = cls._get_history(key)
            if h is None:
                continue
            out[cls.KEY_TO_NAME[key]] = {
                "train_acc": h.get("train_acc", []),
                "val_acc": h.get("val_acc", []),
                "train_loss": h.get("train_loss", []),
                "val_loss": h.get("val_loss", [])
            }
        return out

    @classmethod
    def _load_model(cls, key: str):
        if key in cls._models:
            return cls._models[key]
        try:
            m = ModelFactory.load(key)
            m.eval()
            cls._models[key] = m
            return m
        except Exception as e:
            print(f"[ModelService] Could not load {key}: {e}")
            return None

    @classmethod
    def predict(cls, image: Image.Image, model_key: str = "efficientnetb4") -> dict:
        model = cls._load_model(model_key)
        if model is None:
            raise RuntimeError(
                f"Model '{model_key}' is not available. "
                f"Make sure {model_key}_best.pt exists in {paths.models}."
            )

        img_tensor: torch.Tensor = _EVAL_TRANSFORM(image.convert("RGB"))  # type: ignore[assignment]
        tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)
        t0 = time.time()
        with torch.no_grad():
            out = model(tensor)
            probs_t = torch.softmax(out, dim=1)[0]
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        probs = [round(float(p), 4) for p in probs_t]
        idx = int(probs_t.argmax())

        return {
            "class": Config.CLASS_NAMES[idx],
            "class_index": idx,
            "confidence": round(probs[idx] * 100, 2),
            "probabilities": {
                Config.CLASS_NAMES[i]: round(probs[i] * 100, 2)
                for i in range(len(Config.CLASS_NAMES))
            },
            "inference_ms": elapsed_ms
        }

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": str(Config.DEVICE)})

@app.route("/api/data/stats")
def data_stats():
    return jsonify(DataService.get_distribution_stats())

@app.route("/api/data/samples/<class_name>")
def data_samples(class_name: str):
    n = int(request.args.get("n", 5))
    if class_name not in Config.CLASS_NAMES:
        return jsonify({"error": "Unknown class"}), 400
    imgs = DataService.get_sample_images(class_name, n)
    return jsonify({"images": imgs, "class": class_name})

@app.route("/api/models/metrics")
def model_metrics():
    return jsonify({"models": ModelService.get_metrics()})

@app.route("/api/models/confusion/<model_name>")
def model_confusion(model_name: str):
    cm = ModelService.get_confusion(model_name)
    if cm is None:
        return jsonify({"error": "Confusion matrix not available — run compute_eval.py"}), 404
    return jsonify({**cm, "model": model_name})

@app.route("/api/models/history")
def model_history():
    return jsonify(ModelService.get_history())

@app.route("/api/models/errors/<model_name>")
def model_errors(model_name: str):
    limit = int(request.args.get("limit", 24))
    data = ModelService.get_errors(model_name, limit = limit)
    if data is None:
        return jsonify({"error": "Errors not available — run compute_eval.py"}), 404
    out = []
    for err in data["errors"]:
        path = Path(err["path"])
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB").resize((160, 160))
            buf = io.BytesIO()
            img.save(buf, format = "JPEG", quality = 80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            out.append({
                "thumb": f"data:image/jpeg;base64,{b64}",
                "true":  err["true"],
                "pred":  err["pred"],
                "confidence": err["confidence"],
                "true_prob":  err["true_prob"],
                "filename":   path.name,
            })
        except Exception:
            continue
    return jsonify({
        "model":        model_name,
        "errors":       out,
        "total_errors": data["total_errors"],
        "n_samples":    data["n_samples"],
    })

@app.route("/api/predict", methods = ["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    model_key = request.form.get("model", "efficientnetb4")
    try:
        img = Image.open(f.stream)
        result = ModelService.predict(img, model_key)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("files")
    model_key = request.form.get("model", "efficientnetb4")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    results = []
    for f in files:
        try:
            img = Image.open(f.stream)
            r = ModelService.predict(img, model_key)
            r["filename"] = f.filename
            results.append(r)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug = False, host = "0.0.0.0", port = 5050)