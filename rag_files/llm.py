from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = "You are a helpful assistant that answers questions using the provided context."

PROMPT = PromptTemplate.from_template(
    "{system}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def build_llm(model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    return ChatGroq(model_name=model_name, temperature=0.2)


def generate_answer(llm: ChatGroq, question: str, context: str) -> str:
    message = PROMPT.format(system=SYSTEM_PROMPT, context=context, question=question)
    return llm.invoke(message).content
