# 🔎 RAG Chatbot – FastAPI + Streamlit + Groq

Retrieval‑Augmented Generation (RAG) chatbot that lets you **chat with your own PDFs and text files**.  
Documents are ingested, chunked, embedded, indexed with FAISS, and queried via a FastAPI backend with a Streamlit chat UI on top.

---

## ✨ Features

- **Domain‑agnostic RAG**  
  Ingests any PDFs/TXT from `data/pdf_files/` and `data/text_files/` and answers questions over that corpus.

- **Modular RAG pipeline**
  - Document ingestion with LangChain loaders (PDF + text).
  - Recursive character chunking.
  - SentenceTransformers embeddings.
  - FAISS inner‑product index for similarity search.
  - Custom `RAGRetriever` abstraction and LLM wrapper. 

- **Full‑stack app**
  - **FastAPI** backend exposing:
    - `POST /query` – ask a question, get an answer + supporting contexts.
    - `POST /upload_docs` – upload new PDFs/TXT and rebuild the index.
    - `GET /stats` – see corpus statistics (docs, chunks, sources).
  - **Streamlit** frontend:
    - Chat‑style interface (`st.chat_message`) for conversational queries.
    - Sidebar with corpus overview and document upload.

- **LLM integration (Groq)**
  - Uses Groq’s `llama-3.1-8b-instant` via LangChain’s `ChatGroq` for fast, instruction‑following responses grounded in retrieved context. 

- **Evaluation**
  - Offline evaluation script computing **precision@5** (and recall variants) on a small labeled question set to verify retrieval quality. 

- **Designed for resumes & real projects**
  - Clean, modular structure: `rag_files/` (engine), `api/` (service), `ui/` (UI).
  - Easy to adapt to any domain: research papers, course notes, internal docs, etc.

---

## 🧱 Architecture Overview

      ┌────────────────────┐
      │    Documents       │
      │  (PDF, TXT, …)     │
      └────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Data Ingestion      │
    │  (LangChain loaders) │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │   Chunking           │
    │ Recursive splitter   │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │   Embeddings         │
    │ SentenceTransformers │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐
    │   Vector Store       │
    │  FAISS (IP index)    │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────┐         ┌──────────────────────┐
    │   RAG Retriever      │         │     LLM (Groq)       │
    │  top‑k relevant      │ ─────▶  │ ChatGroq, prompt     │
    └────────┬─────────────┘         └────────┬─────────────┘
             │                                  │
             └─────────────┬────────────────────┘
                           ▼
                  Final Answer (FastAPI → Streamlit UI)


**Back‑end:** `rag_files/` + `api/`  
**Front‑end:** `ui/streamlit_app.py`  

---

## 📂 Project Structure

```text
RAG_Chatbot/
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI app: /query, /upload_docs, /stats
├── ui/
│   └── streamlit_app.py  # Streamlit chat UI + uploads + corpus overview
├── rag_files/
│   ├── __init__.py
│   ├── data_ingestion.py # PDF/TXT loaders
│   ├── chunking.py       # RecursiveCharacterTextSplitter
│   ├── embeddings.py     # EmbeddingManager (SentenceTransformers)
│   ├── vector_store.py   # VectorStore (FAISS IP index, docs)
│   ├── retriever.py      # RAGRetriever
│   ├── llm.py            # ChatGroq wrapper + prompt
│   ├── pipeline.py       # RAGPipeline + get_pipeline + answer_question
│   └── eval.py           # precision@k (and recall) evaluation
├── data/
│   ├── pdf_files/        # user PDFs (gitignored or small samples)
│   └── text_files/       # user TXT files
├── .env.example          # example env file (no secrets)
├── requirements.txt
├── .gitignore
└── README.md


## ⚙️ Setup & Installation

## Setup

1. Clone the repository

'''bash
git clone https://github.com/ritikaugale/RAG_Chatbot.git
cd RAG_Chatbot

2. Create and activate a virtual environment
'''bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install dependencies
'''bash
pip install -r requirements.txt

4. Configure environment variables

# set your Groq key
cp .env.example .env  # if you create an example file
# edit .env and set GROQ_API_KEY=...

5. Prepare data folders 
'''bash
mkdir -p data/pdf_files data/text_files

## 🚀 Running the App
You run the backend and frontend in separate terminals (same venv).

1. Start the FastAPI backend

'''bash
uvicorn api.main:app --reload


2. Start the Streamlit frontend

'''bash
streamlit run ui/streamlit_app.py


## Using the Chatbot

1. Inspect the corpus
In the Streamlit sidebar:
    “Corpus overview” shows:

            Number of documents.
            Number of chunks.
            List of source paths (PDF/TXT filenames). 

2. Upload your own documents

        Use “Upload PDFs or TXT” in the sidebar.
        Select one or more .pdf / .txt files.
        Click “Save uploaded files”.
        The backend saves them into data/pdf_files / data/text_files and rebuilds the index on the next question.

3. Ask questions

    Type a question in the chat input (e.g. “What is the SSD algorithm used for?”).

    The app:

        Embeds your query.
        Retrieves top‑k relevant chunks via FAISS.
        Sends context + question to the Groq LLM.
        Streams the answer back to the chat UI.

4. Use “Show retrieved contexts” to see exactly which snippets were used. 

5. Tune retrieval

        Adjust Top K contexts slider in the sidebar (e.g. from 3 to 8).
        Higher k = more context, potentially better recall but more noise.


## 📊 Evaluation (Precision@k)

1. Prepare a qrels.json as given in example

2. Run evaluation
'''bash
python -m rag_files.eval



## 📄 License

MIT licence


