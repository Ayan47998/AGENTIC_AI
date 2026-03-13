"""Streamlit UI for the PDF RAG Conversational Assistant."""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Conversational Assistant")

# ── Sidebar: Upload & Controls ──────────────────────────────────

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)

        if resp.status_code == 200:
            data = resp.json()
            st.success(
                f"Uploaded **{data['filename']}** — "
                f"{data['pages']} pages, {data['chunks']} chunks"
            )
        else:
            st.error(f"Upload failed: {resp.text}")

    st.divider()

    # Status
    try:
        status = requests.get(f"{API_URL}/status", timeout=5).json()
        st.metric("Chunks in store", status["documents_loaded"])
        st.metric("Chat messages", status["history_length"])
    except requests.ConnectionError:
        st.warning("API not reachable")

    st.divider()

    if st.button("Reset All", type="secondary"):
        requests.post(f"{API_URL}/reset", timeout=10)
        st.session_state.messages = []
        st.rerun()

# ── Chat Interface ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(
                        f"{src['source']} — Page {src['page']} "
                        f"(score: {src['score']:.4f})"
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": prompt},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except Exception as e:
                answer = f"Error communicating with the API: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.caption(
                        f"{src['source']} — Page {src['page']} "
                        f"(score: {src['score']:.4f})"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
