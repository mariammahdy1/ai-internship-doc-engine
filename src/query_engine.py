import os
import atexit
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

gemini_client = genai.Client(api_key=api_key)

qdrant_client = QdrantClient(path="qdrant_storage")
atexit.register(qdrant_client.close)

COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "gemini-embedding-2"

GENERATION_MODEL = "gemini-3.6-flash"

TOP_K = 3


def create_embedding(text: str) -> list[float]:
    """Generate an embedding for a query."""
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def retrieve_chunks(
    question: str,
    top_k: int = TOP_K,
):
    """Retrieve the most relevant document chunks."""

    query_embedding = create_embedding(question)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    ).points

    return results


def build_context(results) -> str:
    """Build a context block containing retrieved chunks."""

    context_parts = []

    for result in results:
        document_title = result.payload["document_title"]
        chunk_id = result.payload["chunk_id"]
        text = result.payload["text"]

        context_parts.append(
            f"[Doc: {document_title}, Chunk #{chunk_id}]\n"
            f"{text}"
        )

    return "\n\n".join(context_parts)


def answer_question(question: str) -> str:
    """Answer a question using only retrieved document context."""

    results = retrieve_chunks(question)

    if not results:
        return (
            "I cannot answer this question based on the "
            "provided documents."
        )

    context = build_context(results)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided document
context.

Do not use outside knowledge.

Every factual claim in your answer must include an inline
citation in this exact format:

[Doc: DOCUMENT TITLE, Chunk #NUMBER]

If the provided context does not contain enough information
to answer the question, reply EXACTLY:

I cannot answer this question based on the provided documents.

User question:
{question}

Provided document context:
{context}
"""

    response = gemini_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return response.text


def main():
    """Run the interactive command-line query interface."""

    print("Document Q&A Engine")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = answer_question(question)

            print("\nAnswer:")
            print(answer)
            print()

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()