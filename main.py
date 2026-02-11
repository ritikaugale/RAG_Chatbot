from rag_files.pipeline import answer_question

if __name__ == "__main__":
    while True:
        q = input("Ask a question (press or 'q' to quit): ")
        if q.lower() in {"q", "quit", "exit"}:
            break
        answer, _ = answer_question(q)
        print("A:", answer)
        print("-" * 40)
