import dotenv
import sys

dotenv.load_dotenv()

from qdrant_client import QdrantClient
from query.retriever import retrieve
from query.generator import generate
import os, logging

QDRANT_HOST = os.getenv("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", default=6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", default="obsidian_notes")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"""
    =================================
    Obsidian Notes RAG Chatbot
    =================================
    Type your question or 'exit' to quit.
    """)

    try:
        while True:
            question = input("\nYou: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                chunks = retrieve(question, q_client)
                logging.debug(f"\n[{len(chunks)} relevant chunks found]")

                print(f"\nAssistant: ", end="", flush=True)
                response = generate(question, chunks)
            except Exception as e:
                logging.error(f"Error processing question: {e}", exc_info=True)
                print("Something went wrong, try again.")

    except KeyboardInterrupt:
        print("\nGoodbye!")
