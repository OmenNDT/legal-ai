import random
from pathlib import Path
from typing import Tuple
from backend.preprocess.loader import ContractLoader
from backend.preprocess.cleaner import TextCleaner
from backend.preprocess.splitter import SentenceSplitter
from backend.extractive.textrank_extractor import TextRankExtractor
from backend.evaluate.reference_builder import ReferenceBuilder

# Xây dataset (input, target) cho fine-tune
# input = câu được TextRank chọn (đã lọc xuống dưới ~1024 token)
# target = các clause gộp từ master_clauses.csv (vai trò "tóm tắt vàng")
class CuadDatasetBuilder:
    def __init__(self, txt_dir: Path, csv_path: Path, top_k_ratio: float = 0.2, seed: int = 42):
        self.loader = ContractLoader(txt_dir)
        self.ref_builder = ReferenceBuilder(csv_path)
        self.cleaner = TextCleaner()
        self.splitter = SentenceSplitter()
        self.extractor = TextRankExtractor(top_k_ratio = top_k_ratio)
        self.seed = seed

    # Build cặp (input, target) cho 1 doc
    def build_one(self, doc_id: str) -> Tuple[str, str]:
        c = self.loader.load_one(doc_id)
        cleaned = self.cleaner.clean(c.raw_text)
        sents = self.splitter.split(cleaned)
        ext = self.extractor.extract(sents)
        target = self.ref_builder.get_reference(doc_id)
        return ext.as_text(), target

    # Build toàn bộ và chia train/val/test
    def build_all(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict:
        ids = self.loader.list_ids()
        rng = random.Random(self.seed)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": ids[:n_train],
            "val": ids[n_train : n_train + n_val],
            "test": ids[n_train + n_val :]
        }
        data = {"train": [], "val": [], "test": []}
        for split, sids in splits.items():
            for did in sids:
                try:
                    inp, tgt = self.build_one(did)
                    if not inp or not tgt:
                        continue
                    data[split].append({"doc_id": did, "input": inp, "target": tgt})
                except Exception:
                    continue
        return data
