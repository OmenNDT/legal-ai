"""Inference script for LoRA PhoBERT trained on Luật Kế toán 2025.

Usage:
    PYTHONPATH=/path/to/legal-ai:$PYTHONPATH python scripts/inference_lora.py

Interactive mode: python scripts/inference_lora.py --interactive
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.common.config import (
    PHOBERT_MODEL, MAX_SEQ_LENGTH, INTENT_LABELS, NER_LABELS,
)
from src.chatbot.intent_classifier import PhoBERTIntentClassifier
from src.chatbot.ner_tagger import PhoBERTNERTagger


def load_trained_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained LoRA checkpoint and build models."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)

    # Load intent classifier
    intent_model = PhoBERTIntentClassifier()
    intent_state = {k.replace("intent_classifier.", ""): v
                    for k, v in checkpoint["model_state_dict"].items()
                    if k.startswith("intent_classifier.")}
    phobert_state = {k.replace("phobert.", ""): v
                     for k, v in checkpoint["model_state_dict"].items()
                     if k.startswith("phobert.")}
    intent_model.phobert.load_state_dict(phobert_state, strict=False)
    intent_model.classifier.load_state_dict(intent_state, strict=False)
    intent_model.to(device)
    intent_model.eval()

    # Load NER tagger
    ner_model = PhoBERTNERTagger()
    ner_state = {k.replace("ner_classifier.", ""): v
                 for k, v in checkpoint["model_state_dict"].items()
                 if k.startswith("ner_classifier.")}
    ner_model.phobert.load_state_dict(phobert_state, strict=False)
    ner_model.classifier.load_state_dict(ner_state, strict=False)
    ner_model.to(device)
    ner_model.eval()

    intent_label2id = checkpoint.get("intent_label2id", {l: i for i, l in enumerate(INTENT_LABELS)})
    intent_id2label = {v: k for k, v in intent_label2id.items()}
    ner_label2id = checkpoint.get("ner_label2id", {l: i for i, l in enumerate(NER_LABELS)})
    ner_id2label = {v: k for k, v in ner_label2id.items()}

    return tokenizer, intent_model, ner_model, intent_id2label, ner_id2label


def predict_intent(text: str, tokenizer, model, intent_id2label, device="cpu"):
    """Predict intent for a question."""
    model.eval()
    encoding = tokenizer(
        text, max_length=MAX_SEQ_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = model.classifier(cls_output)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "intent": intent_id2label.get(pred_idx, "unknown"),
        "confidence": round(confidence, 4),
        "top3": sorted(
            [(intent_id2label.get(i, "unknown"), probs[0, i].item()) for i in range(len(intent_id2label))],
            key=lambda x: -x[1]
        )[:3]
    }


def predict_ner(text: str, tokenizer, model, ner_id2label, device="cpu"):
    """Predict NER tags for a question."""
    model.eval()
    encoding = tokenizer(
        text, max_length=MAX_SEQ_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.phobert(input_ids=input_ids, attention_mask=attention_mask)
        logits = model.classifier(outputs.last_hidden_state)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())
    entities = []
    current_entity = None
    for token, pred_id in zip(tokens, preds):
        if token in ("", "[CLS]", "[SEP]", "<pad>"):
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        label = ner_id2label.get(pred_id, "O")
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity": label[2:], "text": token.replace("Ġ", ""), "tokens": [token]}
        elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
            current_entity["tokens"].append(token)
            current_entity["text"] += token.replace("Ġ", " ")
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Clean up entity text and filter short noise
    cleaned = []
    for e in entities:
        text_clean = e["text"].strip().replace("Ġ", " ").strip()
        if len(text_clean) > 1:
            cleaned.append({"entity": e["entity"], "text": text_clean})

    return cleaned


def search_answer(question: str, qa_dataset_path: str, top_k: int = 3):
    """Simple keyword-based answer retrieval from QA dataset."""
    with open(qa_dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    keywords = set(question.lower().split())
    scored = []
    for item in data:
        q_words = set(item["question"].lower().split())
        score = len(keywords & q_words) / max(len(keywords), 1)
        scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:top_k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--question", "-q", type=str, help="Single question")
    parser.add_argument("--checkpoint", default="data/models/lora_ke_toan/best_model.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    tokenizer, intent_model, ner_model, intent_id2label, ner_id2label = \
        load_trained_model(args.checkpoint, args.device)
    print("Model loaded. Ready for inference.\n")

    qa_path = "data/processed/qa_ke_toan_train.json"

    if args.interactive:
        print("=== LegalAI LoRA Inference (Luật Kế toán 2025) ===")
        print("Gõ câu hỏi hoặc 'exit' để thoát.\n")
        while True:
            try:
                question = input("Hỏi: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue

            intent_result = predict_intent(question, tokenizer, intent_model, intent_id2label, args.device)
            ner_result = predict_ner(question, tokenizer, ner_model, ner_id2label, args.device)
            answers = search_answer(question, qa_path)

            print(f"\n  Intent: {intent_result['intent']} (conf: {intent_result['confidence']:.3f})")
            print(f"  Top-3 intents: {[(l, round(p,3)) for l,p in intent_result['top3']]}")
            print(f"  Entities: {[f'{e['entity']}:{e['text']}' for e in ner_result]}")
            print(f"  Answer candidates:")
            for i, ans in enumerate(answers[:2], 1):
                print(f"    {i}. [{ans['source']}] {ans['answer'][:200]}...")
            print()
    elif args.question:
        question = args.question
        intent_result = predict_intent(question, tokenizer, intent_model, intent_id2label, args.device)
        ner_result = predict_ner(question, tokenizer, ner_model, ner_id2label, args.device)
        answers = search_answer(question, qa_path)

        print(f"Question: {question}")
        print(f"Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.3f})")
        print(f"Top-3 intents: {intent_result['top3']}")
        print(f"Entities: {ner_result}")
        print(f"Answers:")
        for i, ans in enumerate(answers[:3], 1):
            print(f"  {i}. [{ans['source']}]")
            print(f"     {ans['answer'][:300]}")
    else:
        # Demo with sample questions
        demo_questions = [
            "Điều 1 Luật Kế toán quy định gì về phạm vi điều chỉnh?",
            "Đối tượng áp dụng của Luật Kế toán là ai?",
            "Báo cáo tài chính được hiểu như thế nào?",
            "Người làm kế toán có nghĩa vụ gì?",
            "Thủ tục lập chứng từ kế toán như thế nào?",
        ]
        for q in demo_questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            intent = predict_intent(q, tokenizer, intent_model, intent_id2label, args.device)
            ner = predict_ner(q, tokenizer, ner_model, ner_id2label, args.device)
            answers = search_answer(q, qa_path)
            print(f"Intent: {intent['intent']} (conf: {intent['confidence']:.3f})")
            print(f"NER: {[f'{e['entity']}:{e['text']}' for e in ner]}")
            if answers:
                ans = answers[0]
                print(f"Answer: [{ans['source']}] {ans['answer'][:250]}...")


if __name__ == "__main__":
    main()
