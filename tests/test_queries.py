from src.query_engine import answer_question


def run_test(question: str, expected_text: str, expected_citation: str):
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")

    answer = answer_question(question)

    print(f"ANSWER: {answer}")

    if expected_text.lower() in answer.lower():
        print("PASS: Expected information found.")
    else:
        print("FAIL: Expected information NOT found.")

    if expected_citation.lower() in answer.lower():
        print("PASS: Citation found.")
    else:
        print("FAIL: Citation NOT found.")


def main():
    run_test(
        "What is the monthly retainer in the Master Services Agreement?",
        "$18,500",
        "[Doc: Vendor Contract, Chunk #1]",
    )

    run_test(
        "What caused the login failures in incident INC-4022?",
        "Redis",
        "[Doc: Technical Support Log, Chunk #1]",
    )

    run_test(
        "What databases does SyncFlow Pro support?",
        "PostgreSQL",
        "[Doc: Product Operations Guide, Chunk #1]",
    )

    print("\n" + "=" * 70)
    print("TESTING UNSUPPORTED QUESTION")

    question = "Who is the president of Egypt?"
    answer = answer_question(question)

    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")

    expected_fallback = (
        "I cannot answer this question based on the provided documents."
    )

    if answer.strip() == expected_fallback:
        print("PASS: Correct fallback response.")
    else:
        print("FAIL: Incorrect fallback response.")


if __name__ == "__main__":
    main()