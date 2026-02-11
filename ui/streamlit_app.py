import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
QUERY_URL = f"{API_BASE}/query"
UPLOAD_URL = f"{API_BASE}/upload_docs"
STATS_URL = f"{API_BASE}/stats"


@st.cache_data
def fetch_corpus_stats():
    try:
        resp = requests.get(STATS_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


st.set_page_config(page_title="RAG Chatbot 2025", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG Chatbot")
st.caption("Ask questions about the documents in your corpus.")


# Sidebar: corpus overview
stats = fetch_corpus_stats()
st.sidebar.header("Corpus overview")
if stats:
    st.sidebar.write(f"Documents: {stats['num_docs']}")
    st.sidebar.write(f"Chunks: {stats['num_chunks']}")
    st.sidebar.write("Sources:")
    for s in stats["sources"]:
        st.sidebar.caption(f"- {s}")
else:
    st.sidebar.write("Stats unavailable.")

# Sidebar: upload
st.sidebar.header("Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files and st.sidebar.button("Save uploaded files"):
    files_payload = [
        ("files", (f.name, f.read(), f.type or "application/octet-stream"))
        for f in uploaded_files
    ]
    try:
        resp = requests.post(UPLOAD_URL, files=files_payload, timeout=300)
        resp.raise_for_status()
        st.sidebar.success("Files uploaded and index will rebuild on next question.")
        fetch_corpus_stats.clear()
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")


# Sidebar: retrieval settings
st.sidebar.header("Retrieval settings")
top_k = st.sidebar.slider("Top K contexts", 1, 10, 5)


# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input and call backend
if prompt := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    QUERY_URL,
                    json={"question": prompt, "top_k": top_k},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.write(answer)

                with st.expander("Show retrieved contexts"):
                    for i, ctx in enumerate(data.get("contexts", []), start=1):
                        st.markdown(f"**Context {i}:**")
                        st.write(ctx)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error calling backend: {e}")
                answer = f"[Error] {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
