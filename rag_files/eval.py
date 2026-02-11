import json
from pathlib import Path
from typing import List, Dict, Tuple

from .pipeline import get_pipeline


def load_qrels(path: str | Path) -> List[Dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_relevant(chunk_metadata: dict, example: dict) -> bool:
    src_path = chunk_metadata.get("source", "") or ""
    relevant_files = example.get("relevant_sourcefiles", [])

    # match if the filename appears in the source path
    return any(rf in src_path for rf in relevant_files)



def evaluate_retriever(qrels: list[dict], k: int = 5) -> tuple[float, float]:
    pipeline = get_pipeline()
    retriever = pipeline.retriever

    total_prec = 0.0
    total_rec = 0.0
    n = 0

    for ex in qrels:
        question = ex["question"]
        relevant_files = ex.get("relevant_sourcefiles", [])
        num_rel_docs = len(relevant_files) if relevant_files else 1

        retrieved = retriever.retrieve(question, top_k=k)

        relevant_flags = [is_relevant(r.metadata, ex) for r in retrieved]
        num_rel_ret = sum(relevant_flags)

        prec = num_rel_ret / k
        rec = num_rel_ret / num_rel_docs

        total_prec += prec
        total_rec += rec
        n += 1

    if n == 0:
        return 0.0, 0.0

    return total_prec / n, total_rec / n


if __name__ == "__main__":
    qrels_path = Path("eval/qrels.json")
    qrels = load_qrels(qrels_path)
    p5, r5 = evaluate_retriever(qrels, k=5)
    print(f"precision@5 = {p5:.3f}")
    print(f"recall@5    = {r5:.3f}")
