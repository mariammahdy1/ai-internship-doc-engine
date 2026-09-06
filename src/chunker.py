from dataclasses import dataclass
from pathlib import Path

import tiktoken


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


@dataclass
class DocumentChunk:
    document_title: str
    chunk_id: int
    text: str


def get_tokenizer():
    """Return the tokenizer used for chunking documents."""
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    document_title: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    Split a document into overlapping token-based chunks.

    Each chunk contains approximately chunk_size tokens and
    consecutive chunks overlap by the specified number of tokens.
    """
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0
    chunk_id = 1
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        chunk_text_value = tokenizer.decode(chunk_tokens)

        chunks.append(
            DocumentChunk(
                document_title=document_title,
                chunk_id=chunk_id,
                text=chunk_text_value,
            )
        )

        if end == len(tokens):
            break

        start += step
        chunk_id += 1

    return chunks


def chunk_document(file_path: str) -> list[DocumentChunk]:
    """Read a text document and split it into chunks."""
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    return chunk_text(
        text=text,
        document_title=path.stem,
    )


if __name__ == "__main__":
    documents_dir = Path("documents")

    for file_path in sorted(documents_dir.glob("*.txt")):
        chunks = chunk_document(str(file_path))

        print(f"\n{file_path.name}")
        print(f"Chunks created: {len(chunks)}")

        for chunk in chunks:
            tokenizer = get_tokenizer()
            token_count = len(tokenizer.encode(chunk.text))

            print(
                f"  Chunk #{chunk.chunk_id}: "
                f"{token_count} tokens"
            )