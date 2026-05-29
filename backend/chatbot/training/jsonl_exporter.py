import json
import random
from dataclasses import dataclass
from pathlib import Path
import psycopg2
from ..data_pipeline.db_loader import DbConfig

@dataclass
class SplitResult:
    train_count: int
    val_count: int
    test_count: int
    total: int

class JsonlExporter:
    def __init__(self, config: DbConfig) -> None:
        self._cfg = config

    def _connect(self):
        c = self._cfg
        return psycopg2.connect(
            host = c.host, port = c.port,
            dbname = c.db, user = c.user, password = c.password
        )

    def _fetch_records(self, min_similarity: float, validated_only: bool) -> list[dict]:
        conditions = ["similarity_score >= %s"]
        params: list = [min_similarity]
        if validated_only:
            conditions.append("is_validated = TRUE")
        where = " AND ".join(conditions)
        sql = f"SELECT formatted_prompt, target_answer FROM training_samples WHERE {where}"

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [
                {
                    # prompt = input cho model (không có answer), answer = label
                    # full_text = chuỗi đầy đủ theo đúng format flow: "Ngữ cảnh:... | Câu hỏi:... | Trả lời: ..."
                    "prompt": r[0],
                    "answer": r[1],
                    "full_text": r[0] + " " + (r[1] or "")
                }
                for r in cur.fetchall()
            ]

    @staticmethod
    def _write_jsonl(records: list[dict], path: Path) -> None:
        with path.open("w", encoding = "utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii = False) + "\n")

    def export(self, output_path: Path, min_similarity: float = 0.5, validated_only: bool = True) -> int:
        records = self._fetch_records(min_similarity, validated_only)
        self._write_jsonl(records, output_path)
        print(f"[JsonlExporter] Xuất {len(records)} mẫu → {output_path}")
        return len(records)

    def export_split(self, output_dir: Path, min_similarity: float = 0.5, validated_only: bool = True, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42) -> SplitResult:
        assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-6
        test_ratio = round(1.0 - train_ratio - val_ratio, 10)
        records = self._fetch_records(min_similarity, validated_only)
        total = len(records)
        random.seed(seed)
        random.shuffle(records)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = total - n_train - n_val

        splits = {
            "train": records[:n_train],
            "val": records[n_train : n_train + n_val],
            "test": records[n_train + n_val :]
        }

        output_dir.mkdir(parents = True, exist_ok = True)
        for name, subset in splits.items():
            path = output_dir / f"{name}.jsonl"
            self._write_jsonl(subset, path)
            print(f"[JsonlExporter] {name:5s}: {len(subset):>6,} mẫu → {path}")

        print(
            f"[JsonlExporter] Tổng: {total:,} | "
            f"train={n_train:,} ({train_ratio:.0%}) | "
            f"val={n_val:,} ({val_ratio:.0%}) | "
            f"test={n_test:,} ({test_ratio:.0%})"
        )
        return SplitResult(
            train_count = n_train,
            val_count = n_val,
            test_count = n_test,
            total = total
        )