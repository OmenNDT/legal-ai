"""Streamlit UI for LegalAI Platform — 4 module demo in 1 interface."""

import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

API_URL = "http://localhost:9000"

st.set_page_config(page_title="LegalAI", page_icon="⚖️", layout="wide")
st.title("⚖️ LegalAI — Trợ lý Pháp luật Thông minh")
st.caption("4 mô-đun: Chatbot (NLP) → Tìm kiếm (TT) → Tóm tắt (Python ML) → Đồ thị kiến thức (AI)")

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    api_status = st.session_state.get("api_status", "unknown")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        api_status = "✅ Connected" if resp.status_code == 200 else "❌ Error"
    except Exception:
        api_status = "❌ Offline"
    st.write(f"API: {api_status}")

    st.markdown("---")
    st.markdown("""
    **Mô-đun:**
    1. 🤖 Chatbot — Hỏi đáp pháp luật (NLP)
    2. 🔍 Tìm kiếm — BM25 + Trie (Thiết kế TT)
    3. 📝 Tóm tắt — PhoBERT scoring (Python ML)
    4. 🕸️ Đồ thị — Knowledge graph (AI)
    """)

# ── Main tabs ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Hỏi đáp", "🔍 Tìm kiếm", "📝 Tóm tắt", "🕸️ Đồ thị kiến thức"])

# ── Tab 1: Chatbot ─────────────────────────────────────
with tab1:
    st.header("🤖 Hỏi đáp pháp luật")
    question = st.text_input("Nhập câu hỏi:", placeholder="Thông tư 99 quy định gì về chế độ kế toán?")

    if st.button("Hỏi", key="chat_btn") and question:
        with st.spinner("Đang xử lý..."):
            try:
                resp = requests.post(f"{API_URL}/chat", json={"question": question, "top_k": 5}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(f"**Trả lời:** {data['answer']}")
                    st.caption(f"Intent: {data['intent']} (độ tin cậy: {data['confidence']:.2f})")

                    if data.get("entities"):
                        st.markdown("**Thực thể nhận diện:**")
                        for ent in data["entities"]:
                            st.markdown(f"- {ent.get('entity', '')}: {ent.get('text', '')}")

                    if data.get("summary"):
                        st.markdown("**Tóm tắt:**")
                        st.info(data["summary"])

                    if data.get("reasoning"):
                        st.markdown("**Luận lý:**")
                        for step in data["reasoning"]:
                            st.markdown(f"- {step}")
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except requests.ConnectionError:
                st.error("Không kết nối được API. Chạy: `python -m uvicorn src.app:app --port 9000`")

# ── Tab 2: Search ─────────────────────────────────────
with tab2:
    st.header("🔍 Tìm kiếm văn bản pháp luật")
    query = st.text_input("Truy vấn:", placeholder="chế độ kế toán doanh nghiệp")
    top_k = st.slider("Số kết quả:", 1, 20, 5)

    if st.button("Tìm kiếm", key="search_btn") and query:
        with st.spinner("Đang tìm..."):
            try:
                resp = requests.post(f"{API_URL}/search", json={"query": query, "top_k": top_k}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        for i, r in enumerate(results, 1):
                            st.markdown(f"**{i}.** {r.get('title', r.get('doc_id', 'N/A'))}")
                            st.markdown(f"   Score: {r.get('score', 0):.4f}")
                            if r.get("type"):
                                st.caption(f"   Loại: {r['type']}")
                    else:
                        st.warning("Không tìm thấy kết quả.")
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except requests.ConnectionError:
                st.error("Không kết nối được API.")

    # Autocomplete demo
    st.markdown("---")
    prefix = st.text_input("Gợi ý tự động:", placeholder="Thông tư...")
    if prefix and len(prefix) >= 2:
        try:
            resp = requests.post(f"{API_URL}/search/autocomplete?prefix={prefix}", timeout=5)
            if resp.status_code == 200:
                suggestions = resp.json().get("suggestions", [])
                for s in suggestions[:5]:
                    st.markdown(f"- {s['word']} (tf={s.get('freq', '?')})")
        except Exception:
            pass

# ── Tab 3: Summarize ───────────────────────────────────
with tab3:
    st.header("📝 Tóm tắt văn bản pháp luật")
    doc_text = st.text_area("Văn bản:", height=200, placeholder="Dán văn bản pháp luật vào đây...")
    summary_query = st.text_input("Câu hỏi liên quan (tùy chọn):", placeholder="Quy định gì về kế toán?")
    num_sentences = st.slider("Số câu tóm tắt:", 1, 10, 4)

    if st.button("Tóm tắt", key="sum_btn") and doc_text:
        with st.spinner("Đang tóm tắt..."):
            try:
                resp = requests.post(f"{API_URL}/summarize", json={
                    "document": doc_text,
                    "query": summary_query,
                    "top_k": num_sentences,
                }, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown("**Tóm tắt:**")
                    st.success(data.get("summary", "Không thể tạo tóm tắt."))

                    # Show score breakdown if available
                    if "scores" in data:
                        with st.expander("Chi tiết điểm câu"):
                            for idx in data.get("selected_indices", []):
                                st.markdown(f"- Câu {idx}")
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except requests.ConnectionError:
                st.error("Không kết nối được API.")

# ── Tab 4: Knowledge Graph ─────────────────────────────
with tab4:
    st.header("🕸️ Đồ thị kiến thức pháp luật")
    query_type = st.selectbox("Loại truy vấn:", ["validity", "amendments", "related", "path"])
    doc_id = st.text_input("ID văn bản:", placeholder="THONG_TU_99/2015/TT-BTC")
    target_id = st.text_input("ID đích (cho path):", placeholder="LUAT_Ke_toan") if query_type == "path" else None

    if st.button("Truy vấn", key="kg_btn") and doc_id:
        with st.spinner("Đang truy vấn đồ thị..."):
            try:
                payload = {"doc_id": doc_id, "query_type": query_type}
                if target_id:
                    payload["target_id"] = target_id
                resp = requests.post(f"{API_URL}/knowledge/query", json=payload, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(f"**Câu hỏi:** {data.get('question', '')}")
                    st.markdown(f"**Trả lời:** {data.get('answer', '')}")
                    st.caption(f"Độ tin cậy: {data.get('confidence', 0):.2f}")

                    if data.get("reasoning_steps"):
                        st.markdown("**Luận lý:**")
                        for step in data["reasoning_steps"]:
                            st.markdown(f"- {step}")

                    if data.get("evidence"):
                        st.markdown("**Bằng chứng:**")
                        for ev in data["evidence"][:5]:
                            st.json(ev)
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except requests.ConnectionError:
                st.error("Không kết nối được API.")

    # Stats
    st.markdown("---")
    if st.button("Thống kê đồ thị"):
        try:
            resp = requests.get(f"{API_URL}/knowledge/stats", timeout=5)
            if resp.status_code == 200:
                st.json(resp.json())
        except Exception:
            st.error("Không kết nối được API.")