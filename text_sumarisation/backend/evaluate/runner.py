from typing import List, Optional
from tqdm import tqdm
from backend.config.settings import get_settings
from backend.preprocess.loader import ContractLoader
from backend.hybrid.pipeline import HybridPipeline
from backend.evaluate.reference_builder import ReferenceBuilder
from backend.evaluate.rouge_scorer import RougeEvaluator
from backend.extractive.label_guided import LabelGuidedReranker
from backend.utils.io import JsonIO

# Chạy toàn bộ benchmark trên 510 file và lưu kết quả ROUGE
class EvalRunner:
    def __init__(self, extractor_name: str = "textrank", use_abstractive: bool = True):
        self.settings = get_settings()
        self.loader = ContractLoader(self.settings.TXT_DIR)
        self.ref_builder = ReferenceBuilder(self.settings.CSV_FILE)
        self.rouge = RougeEvaluator()
        reranker = None
        if self.settings.LABEL_GUIDED:
            reranker = LabelGuidedReranker(
                reference_builder = self.ref_builder,
                sbert_model = self.settings.SBERT_MODEL,
                device = self.settings.device(),
                sim_threshold = self.settings.LABEL_SIM_THRESHOLD,
                max_extra_ratio = self.settings.LABEL_MAX_EXTRA_RATIO
            )
        self.pipeline = HybridPipeline(self.settings, extractor_name = extractor_name, use_abstractive = use_abstractive, reranker = reranker)

    # Tuỳ chọn limit để chạy POC nhanh
    def run(self, limit: Optional[int] = None, doc_ids: Optional[List[str]] = None) -> dict:
        ids = doc_ids if doc_ids else self.loader.list_ids()
        if limit:
            ids = ids[:limit]
        per_doc = []
        preds, refs = [], []
        for did in tqdm(ids, desc = "Eval"):
            try:
                contract = self.loader.load_one(did)
                ref = self.ref_builder.get_reference(did)
                result = self.pipeline.run(contract.raw_text, doc_id = did)
                pred = result.abstractive.text if result.abstractive else result.extractive.as_text()
                rouge = self.rouge.score(pred, ref)
                per_doc.append({
                    "doc_id": did,
                    "rouge": rouge.to_dict(),
                    "timings": result.timings
                })
                preds.append(pred)
                refs.append(ref)
            except Exception as e:
                per_doc.append({"doc_id": did, "error": str(e)})
        avg, _ = self.rouge.score_batch(preds, refs)
        out = {
            "extractor": self.pipeline.extractor.name,
            "use_abstractive": self.pipeline.use_abstractive,
            "num_docs": len(per_doc),
            "average": avg.to_dict(),
            "per_doc": per_doc,
        }
        out_path = self.settings.OUT_EVAL / f"rouge_{self.pipeline.extractor.name}_{'abs' if self.pipeline.use_abstractive else 'ext'}.json"
        JsonIO.write(out_path, out)
        return out
