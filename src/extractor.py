from pydantic import BaseModel, field_validator
from enum import Enum
from google import genai
from dotenv import load_dotenv
import os
import re
import time
from pathlib import Path
import json


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class Category(str, Enum):
    CONTRACT = "Contract"
    TECHNICAL_SPEC = "Technical Spec"
    SUPPORT_LOG = "Support Log"
    PRODUCT_GUIDE = "Product Guide"


class UrgencyRating(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float


class DocumentMetadata(BaseModel):
    document_title: str
    category: Category
    executive_summary: str
    key_entities: list[str]
    urgency_rating: UrgencyRating

    @field_validator("executive_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Executive summary cannot be empty.")

        cleaned_value = re.sub(
            r"\b(?:LLC|Ltd|Inc|Dr|Mr|Ms)\.",
            lambda match: match.group(0).replace(".", ""),
            value,
        )

        cleaned_value = re.sub(
            r"\bv\d+\.\d+\.\d+",
            lambda match: match.group(0).replace(".", ""),
            cleaned_value,
        )

        sentences = re.split(r"(?<=[.!?])\s+", cleaned_value)

        if len(sentences) > 3:
            raise ValueError(
                "Executive summary must contain at most 3 sentences."
            )

        return value


def read_document(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_metadata(
    document_text: str,
) -> tuple[DocumentMetadata, UsageInfo]:
    prompt = f"""
    Analyze the following business document and extract its metadata.

    Instructions:
    - document_title: Identify the title or main identifying name of the document.
    - category: Classify the document as exactly one of: Contract, Technical Spec, Support Log, Product Guide.
    - executive_summary: Summarize the most important information in no more than 3 sentences.
    - key_entities: Extract the most important entities from the document.
    Include named people, companies, organizations, systems, products,
    technologies, identifiers, dates, monetary values, percentages,
    contract terms, and other significant factual entities.
    Return them as a list of strings.
    - urgency_rating: Classify the urgency as exactly one of: High, Medium, Low.
      - High: Critical incidents, outages, security issues, or matters requiring immediate attention.
      - Medium: Important issues or actions that require attention but are not immediately critical.
      - Low: Routine information, normal operations, or matters that do not require urgent action.

    Base your answer only on information contained in the document. Do not invent facts.
    Preserve important entities exactly as they appear in the document, including
    dates, monetary values, percentages, identifiers, and organization names.

    Document:
    {document_text}
    """

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DocumentMetadata,
                },
            )

            metadata = DocumentMetadata.model_validate_json(
                response.text
            )

            usage = response.usage_metadata

            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0

            input_cost_per_million = 0.0
            output_cost_per_million = 0.0

            estimated_cost = (
                (input_tokens / 1_000_000)
                * input_cost_per_million
                + (output_tokens / 1_000_000)
                * output_cost_per_million
            )

            usage_info = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
            )

            return metadata, usage_info

        except Exception as error:
            if attempt == max_retries - 1:
                raise error

            delay = 5 * (2 ** attempt)

            print(
                f"Extraction failed. Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError("Metadata extraction failed.")


def process_document(
    file_path: str,
) -> tuple[DocumentMetadata, UsageInfo]:
    document_text = read_document(file_path)
    return extract_metadata(document_text)


def process_all_documents(
    folder_path: str,
) -> list[tuple[str, DocumentMetadata, UsageInfo]]:
    folder = Path(folder_path)
    results = []

    for file_path in folder.glob("*.txt"):
        try:
            metadata, usage_info = process_document(str(file_path))
            results.append(
                (file_path.name, metadata, usage_info)
            )
        except Exception as error:
            print(f"Failed to process {file_path.name}: {error}")

    return results


def main():
    results = process_all_documents("documents")

    metadata_list = [metadata.model_dump() for _, metadata, _ in results]

    usage_list = [
        {
            "document": filename,
            **usage.model_dump(),
        }
        for filename, _, usage in results
    ]

    with open("output.json", "w", encoding="utf-8") as file:
        json.dump(metadata_list, file, indent=4)

    with open("usage.json", "w", encoding="utf-8") as file:
        json.dump(usage_list, file, indent=4)

    print("Metadata successfully saved to output.json")
    print("Usage information successfully saved to usage.json")


if __name__ == "__main__":
    main()

