import argparse
from pathlib import Path
from dotenv import load_dotenv
from ..data_pipeline.embedder import Embedder
from ..data_pipeline.db_loader import DbConfig
from .qa_mapper import QaLoader, VectorSearcher, TrainingSampleSaver, QaMapper
from .jsonl_exporter import JsonlExporter

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

def _build_mapper(top_k: int = 2, min_sim: float = 0.5) -> QaMapper:
    cfg = DbConfig.from_env()
    return QaMapper(
        qa_loader = QaLoader(cfg),
        embedder = Embedder(),
        searcher = VectorSearcher(cfg, top_k = top_k),
        saver = TrainingSampleSaver(cfg),
        min_similarity = min_sim
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Phase 2 — Training Dataset Builder")
    sub = parser.add_subparsers(dest = "cmd", required = True)
    p_map = sub.add_parser("map", help = "Map Q&A → training_samples")
    p_map.add_argument("--limit", type = int, default = None, help = "Số lượng Q&A tối đa")
    p_map.add_argument("--top-k", type = int, default = 2, help = "Số chunk lấy mỗi câu hỏi")
    p_map.add_argument("--min-sim", type = float, default = 0.5, help = "Ngưỡng cosine tối thiểu")
    p_exp = sub.add_parser("export", help = "Xuất training_samples → một file JSONL")
    p_exp.add_argument("--output", type = Path, required = True, help = "Đường dẫn file .jsonl đầu ra")
    p_exp.add_argument("--min-sim", type = float, default = 0.5)
    p_exp.add_argument("--all", action = "store_true", help = "Xuất cả mẫu chưa validate")
    p_spl = sub.add_parser("split", help = "Xuất train.jsonl / val.jsonl / test.jsonl (70/15/15)")
    p_spl.add_argument("--output-dir", type = Path,  required = True, help = "Thư mục chứa 3 file đầu ra")
    p_spl.add_argument("--min-sim", type = float, default = 0.5)
    p_spl.add_argument("--all", action = "store_true", help = "Xuất cả mẫu chưa validate")
    p_spl.add_argument("--train-ratio", type = float, default = 0.70, help = "Tỉ lệ train (mặc định 0.70)")
    p_spl.add_argument("--val-ratio", type = float, default = 0.15, help = "Tỉ lệ val   (mặc định 0.15)")
    p_spl.add_argument("--seed", type = int, default = 42, help = "Random seed shuffle")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "map":
        mapper = _build_mapper(top_k = args.top_k, min_sim = args.min_sim)
        mapper.run(limit = args.limit)

    elif args.cmd == "export":
        cfg = DbConfig.from_env()
        exporter = JsonlExporter(cfg)
        exporter.export(
            output_path = args.output,
            min_similarity = args.min_sim,
            validated_only = not args.all
        )

    elif args.cmd == "split":
        cfg = DbConfig.from_env()
        exporter = JsonlExporter(cfg)
        exporter.export_split(
            output_dir = args.output_dir,
            min_similarity = args.min_sim,
            validated_only = not args.all,
            train_ratio = args.train_ratio,
            val_ratio = args.val_ratio,
            seed = args.seed
        )
