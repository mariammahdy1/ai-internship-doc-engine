import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from chunker import chunk_document


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

gemini_client = genai.Client(api_key=api_key)

qdrant_client = QdrantClient(path="qdrant_storage")

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "gemini-embedding-2"
VECTOR_SIZE = 3072


def create_collection():
    """Create the Qdrant collection if it does not already exist."""

    existing_collections = qdrant_client.get_collections().collections

    if COLLECTION_NAME not in [collection.name for collection in existing_collections]:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def create_embedding(text: str) -> list[float]:
    """Generate an embedding for a piece of text using Gemini."""

    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def index_documents(documents_dir: str = "documents"):
    """Chunk all documents, embed the chunks, and store them in Qdrant."""

    create_collection()

    points = []
    point_id = 1

    for file_path in sorted(Path(documents_dir).glob("*.txt")):
        chunks = chunk_document(str(file_path))

        for chunk in chunks:
            embedding = create_embedding(chunk.text)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_title": chunk.document_title,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                    },
                )
            )

            point_id += 1

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    print(f"Indexed {len(points)} document chunks.")


if __name__ == "__main__":
    index_documents()