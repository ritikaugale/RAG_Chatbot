from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from .config import PDF_DIR, TEXT_DIR

def load_text_documents(text_dir: Path | None = None) -> list[Document]:
    text_dir = text_dir or TEXT_DIR
    loader = DirectoryLoader(
        str(text_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=False,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()

def load_pdf_documents(pdf_dir: Path | None = None) -> list[Document]:
    pdf_dir = pdf_dir or PDF_DIR
    loader = DirectoryLoader(
        str(pdf_dir),
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False,
    )
    return loader.load()

def load_all_documents() -> list[Document]:
    docs = load_text_documents() + load_pdf_documents()
    return docs
