# Automated Document Intelligence & Query Engine

An AI-powered document processing system developed for the AI internship project.

The system ingests business documents, extracts structured metadata using Google Gemini, validates the output using Pydantic, and records token usage and estimated API costs.

## Features

### Milestone 1

- Automated ingestion of plain-text documents
- AI-powered metadata extraction using Google Gemini
- Strict structured JSON output
- Pydantic schema validation
- Document categorization
- Executive summaries limited to three sentences
- Key entity extraction
- Urgency classification
- Retry handling for extraction and validation failures
- Input, output, and total token tracking
- Estimated API cost tracking
- JSON output export

### Milestone 2

Milestone 2 extends the system with document chunking, embeddings, vector search, and grounded question answering.

## Project Structure

```text
ai-internship-doc-engine/
│
├── documents/
│   ├── Technical Support Log.txt
│   ├── Vendor Contract.txt
│   └── Product Operations Guide.txt
│
├── src/
│   ├── extractor.py
│   ├── chunker.py
│   ├── vector_store.py
│   └── query_engine.py
│
├── .env
├── .gitignore
├── output.json
├── usage.json
└── README.md


## Milestone 2 - Document Q&A

Milestone 2 adds a retrieval-augmented generation pipeline.

The system:

1. Splits documents into overlapping chunks.
2. Generates embeddings using Google Gemini.
3. Stores embeddings in a local Qdrant vector database.
4. Retrieves the top 3 relevant chunks for each question.
5. Passes the retrieved context to Gemini.
6. Generates an answer using only the retrieved documents.
7. Includes document and chunk citations with factual claims.
8. Returns a fallback response when the documents do not contain enough information.

### Running the Q&A Engine

From the project root:

```bash
python src\query_engine.py

## Testing

The project includes tests for the document retrieval and question-answering pipeline.

Run:

```bash
python tests\test_queries.py