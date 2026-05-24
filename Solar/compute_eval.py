"""
Generate <model>_eval.json beside each .pt checkpoint.
Loads the existing .pt files (no re-training), runs the test split through each model,
and writes accuracy/precision/recall/f1/auc/confusion_matrix/per_class_metrics +
inference_ms benchmark + error samples for the dashboard.
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

from config.GetPath import paths
from utils.config import Config
from services.data_module import SolarPanelDataModule
from services.model_factory import ModelFactory
from services.evaluator import Evaluator

logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_eval")

MODELS = ["efficientnetb4", "resnet50", "vit"]


def main() -> int:
    Config.seed_everything()
    log.info("Device: %s", Config.DEVICE)

    dm = SolarPanelDataModule(
        img_size = Config.IMG_SIZE,
        batch_size = Config.BATCH_SIZE,
        seed = Config.SEED
    )
    dm.setup()
    loaders = dm.get_loaders(num_workers = 2)
    test_loader = loaders["test"]

    test_subset = dm.test_dataset
    assert test_subset is not None
    base_ds = test_subset.dataset
    test_indices = list(test_subset.indices)
    file_paths = [str(Path(base_ds.samples[i][0]).resolve()) for i in test_indices]

    evaluator = Evaluator(device = Config.DEVICE, class_names = Config.CLASS_NAMES)

    for model_name in MODELS:
        pt_path = paths.models / f"{model_name}_best.pt"
        if not pt_path.exists():
            log.warning("Skipping %s — checkpoint not found at %s", model_name, pt_path)
            continue

        log.info("[%s] Loading checkpoint...", model_name)
        model = ModelFactory.load(model_name)

        log.info("[%s] Predicting on test set (%d samples)...", model_name, len(test_subset))
        y_true, y_pred, y_prob = evaluator.predict(model, test_loader)

        report = evaluator.full_report(y_true, y_pred, y_prob)
        log.info(
            "[%s] acc=%.4f f1=%.4f auc=%s n=%d",
            model_name, report["accuracy"], report["f1"],
            f"{report['auc']:.4f}" if report["auc"] is not None else "N/A",
            report["n_samples"],
        )

        log.info("[%s] Benchmarking inference latency...", model_name)
        report["inference_ms"] = evaluator.benchmark_inference_ms(
            model, img_size = Config.IMG_SIZE, n_warmup = 5, n_runs = 30
        )
        log.info("[%s] inference_ms=%.2f", model_name, report["inference_ms"])

        errors = []
        for i, (yt, yp) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            if yt != yp:
                errors.append({
                    "path": file_paths[i],
                    "true": Config.CLASS_NAMES[yt],
                    "pred": Config.CLASS_NAMES[yp],
                    "confidence": float(y_prob[i][yp]),
                    "true_prob":  float(y_prob[i][yt]),
                })
        report["errors"] = errors
        log.info("[%s] %d misclassified samples", model_name, len(errors))

        out_path = paths.models / f"{model_name}_eval.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent = 2)
        log.info("[%s] Saved -> %s", model_name, out_path)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
