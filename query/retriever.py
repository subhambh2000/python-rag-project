from qdrant_client import QdrantClient
from pipeline.embedder import embed_one
import os

QDRANT_HOST = os.getenv("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", default=6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", default="obsidian_notes")
TOP_K = 5
SCORE_THRESHOLD = float(os.getenv("THRESHOLD", default=0.35))


def retrieve(
        query: str,
        client: QdrantClient,
        top_k: int = TOP_K,
        threshold: float = SCORE_THRESHOLD
) -> list[dict]:
    embedded_query = embed_one(query)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedded_query,  # type: ignore
        limit=top_k,
        score_threshold=threshold,
        with_payload=True
    )

    query_results = [
        {
            "content": point.payload["content"],
            "source_file": point.payload["source_file"],
            "header_path": point.payload["header_path"],
            "folder": point.payload["folder"],
            "score": point.score
        }
        for point in response.points
    ]

    seen = set()
    deduplicated = []

    for result in query_results:
        key = result["header_path"] + result["source_file"]
        if key not in seen:
            seen.add(key)
            deduplicated.append(result)

    return deduplicated


if __name__ == "__main__":
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    query_1 = "what are large cap funds"
    query_2 = "what are debt funds"
    query_3 = "how to evaluate a stock"

    result_1 = retrieve(query_1, client)
    result_2 = retrieve(query_2, client)
    result_3 = retrieve(query_3, client)

    print(f"Query: {query_1}")
    for point in result_1:
        print(f"{point["score"]}\t{point["header_path"]} ({point["source_file"]}) \n {point["content"][:150]}")

    print(f"\nQuery: {query_2}")
    for point in result_2:
        print(f"{point["score"]}\t{point["header_path"]} ({point["source_file"]}) \n {point["content"][:150]}")

    print(f"\nQuery: {query_3}")
    for point in result_3:
        print(f"{point["score"]}\t{point["header_path"]} ({point["source_file"]}) \n {point["content"][:150]}")
