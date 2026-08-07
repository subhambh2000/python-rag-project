import os, dotenv

dotenv.load_dotenv()

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

from chunker import chunk_folder
from embedder import embed
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_SIZE = 100

QDRANT_HOST = os.getenv("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", default=6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", default="obsidian_notes")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", default=384))

def create_collection(client: QdrantClient):
    if client.collection_exists(COLLECTION_NAME):
        logging.debug(f"Collection {COLLECTION_NAME} already exists.")
    else:
        logging.debug(f"Creating collection {COLLECTION_NAME}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        logging.debug(f"Collection {COLLECTION_NAME} created.")


def ingest(notes_folder: str):
    logging.debug(f"Chunking folder: {notes_folder}")
    chunks = chunk_folder(notes_folder)
    logging.debug(f"Chunked into {len(chunks)} total chunks")

    contents = [chunk["content"] for chunk in chunks]

    logging.debug(f"Embedding {len(contents)} chunk contents...")
    vectors = embed(contents)
    logging.debug(f"Embedding complete. Vector size: {len(vectors[0]) if vectors else 0}")

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload={
                "content": chunk["content"],
                "source_file": chunk["source_file"],
                "folder": chunk["folder"],
                "header_path": chunk["header_path"],
                "char_count": chunk["char_count"]
            }
        ) for chunk, vector in zip(chunks, vectors)]

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logging.debug(f"Upserted batch: {i // BATCH_SIZE + 1} ({len(batch)} points)")


if __name__ == "__main__":
    logging.debug("Starting ingestion pipeline...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        create_collection(client)
        ingest("data/notes")
        logging.debug("Ingestion Complete")
    except Exception as e:
        logging.error(f"Ingestion failed: {e}", exc_info=True)
