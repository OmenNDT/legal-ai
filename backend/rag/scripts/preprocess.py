"""Preprocess legal data: segment, clean, split train/val/test."""

import json
import random
from pathlib import Path

from src.common.config import (
    QA_DATA_PATH, LEGAL_DATA_PATH, PROCESSED_DIR,
    INTENT_LABELS, NER_LABELS,
)
from src.common.text_processor import (
    clean_vietnamese, segment_text, segment_sentences,
    extract_legal_references, build_training_labels_for_ner,
)


def preprocess_qa_data():
    """Preprocess Q&A data for intent classification and NER training."""
    print("Loading Q&A data...")
    with open(QA_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: {"count": N, "sample": {...}} and [item, ...]
    if isinstance(data, dict) and "sample" in data:
        items = [_unwrap(data["sample"])]
        # yuiTC format: keys are string indices "0", "1", ..., each is a QA pair
        for key in data:
            if isinstance(data[key], dict) and "question" in data[key]:
                items.append(_unwrap(data[key]))
    elif isinstance(data, list):
        items = [_unwrap(i) for i in data]
    else:
        print(f"Unexpected data format: {type(data)}")
        return

    print(f"  Loaded {len(items)} Q&A pairs")

    # Segment and clean
    processed = []
    for item in items:
        question = clean_vietnamese(item["question"])
        context = clean_vietnamese(item["context"])

        # Extract NER labels from context
        references = extract_legal_references(context)
        ner_labels = build_training_labels_for_ner(context, references)

        # Segment for PhoBERT
        seg_question = segment_text(question)
        seg_context = segment_text(context)

        processed.append({
            "question": question,
            "context": context,
            "segmented_question": seg_question,
            "segmented_context": seg_context,
            "references": references,
            "ner_labels": ner_labels,
            "qid": item.get("qid", ""),
            "cid": item.get("cid", ""),
        })

    # Split train/val/test (80/10/10)
    random.shuffle(processed)
    n = len(processed)
    train = processed[: int(n * 0.8)]
    val = processed[int(n * 0.8) : int(n * 0.9)]
    test = processed[int(n * 0.9) :]

    # Save
    for split, data_list in [("train", train), ("val", val), ("test", test)]:
        out_path = PROCESSED_DIR / f"qa_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"  Saved {split}: {len(data_list)} items → {out_path}")


def preprocess_legal_docs():
    """Preprocess legal document data for search index and knowledge graph."""
    print("Loading legal document data...")
    with open(LEGAL_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        docs = data
    elif isinstance(data, dict):
        # Try to find the document list
        for key in data:
            if isinstance(data[key], list) and len(data[key]) > 0:
                docs = data[key]
                break
        else:
            print(f"  Could not find document list in data")
            return
    else:
        print(f"Unexpected data format: {type(data)}")
        return

    print(f"  Loaded {len(docs)} documents")

    # Process each document
    processed = []
    for doc in docs:
        content = doc.get("content", doc.get("text", ""))
        if not content:
            continue

        content = clean_vietnamese(content)
        segmented = segment_text(content)
        sentences = segment_sentences(content)
        references = extract_legal_references(content)

        processed.append({
            "id": doc.get("id", ""),
            "title": doc.get("title", ""),
            "type": doc.get("type", doc.get("law_type", "")),
            "content": content,
            "segmented_content": segmented,
            "sentences": sentences,
            "references": references,
            "law_number": doc.get("law_number", ""),
            "issuer": doc.get("issuer", ""),
            "effective_date": doc.get("effective_date", ""),
        })

    # Save processed documents
    out_path = PROCESSED_DIR / "legal_docs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(processed)} documents → {out_path}")


def _unwrap(entry: dict) -> dict:
    """Normalize a QA entry into consistent format."""
    return {
        "question": entry.get("question", ""),
        "context": entry.get("context_list", entry.get("context", "")),
        "qid": entry.get("qid", ""),
        "cid": entry.get("cid", ""),
    }


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if QA_DATA_PATH.exists():
        preprocess_qa_data()
    else:
        print(f"Q&A data not found at {QA_DATA_PATH}. Run download_data.py first.")

    if LEGAL_DATA_PATH.exists():
        preprocess_legal_docs()
    else:
        print(f"Legal data not found at {LEGAL_DATA_PATH}. Run download_data.py first.")