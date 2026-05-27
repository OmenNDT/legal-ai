import argparse
from pathlib import Path
from backend.config.settings import get_settings
from backend.training.dataset_builder import CuadDatasetBuilder
from backend.training.trainer import BartFineTuner
from backend.utils.logger import Logger
from backend.utils.io import JsonIO

# Entry point chạy fine-tune trên worker1 (GPU)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 3)
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--grad_accum", type = int, default = 8)
    parser.add_argument("--lr", type = float, default = 3e-5)
    parser.add_argument("--use_lora", action = "store_true")
    parser.add_argument("--no_fp16", action = "store_true")
    parser.add_argument("--rebuild_dataset", action = "store_true")
    parser.add_argument("--output_dir", type = str, default = "outputs/bart-cuad")
    parser.add_argument("--early_stopping_patience", type = int, default = 2)
    parser.add_argument("--early_stopping_threshold", type = float, default = 0.0)
    args = parser.parse_args()

    settings = get_settings()
    logger = Logger.setup(settings.LOG_DIR, name = "train")
    logger.info("Bắt đầu fine-tune BART")
    logger.info(f"Device được phát hiện: {settings.device()}")

    # Cache dataset đã build vào outputs/eval để không build lại
    cache_path = settings.OUT_EVAL / "dataset_cuad.json"
    if cache_path.exists() and not args.rebuild_dataset:
        data = JsonIO.read(cache_path)
        logger.info(f"Đọc dataset từ cache: {cache_path}")
    else:
        builder = CuadDatasetBuilder(settings.TXT_DIR, settings.CSV_FILE, top_k_ratio = settings.TOP_K_RATIO)
        data = builder.build_all()
        JsonIO.write(cache_path, data)
        logger.info(f"Đã build dataset: train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")

    tuner = BartFineTuner(
        model_name = settings.BART_MODEL,
        output_dir = Path(args.output_dir),
        max_input = settings.BART_MAX_INPUT,
        max_target = settings.BART_MAX_OUTPUT,
        epochs = args.epochs,
        batch_size = args.batch_size,
        grad_accum = args.grad_accum,
        lr = args.lr,
        fp16 = not args.no_fp16,
        use_lora = args.use_lora,
        early_stopping_patience = args.early_stopping_patience,
        early_stopping_threshold = args.early_stopping_threshold,
    )
    final_path = tuner.fit(data)
    logger.info(f"Đã lưu model fine-tune tại: {final_path}")

if __name__ == "__main__":
    main()
