import logging

import groq
import os

from groq.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from qdrant_client import QdrantClient

from query.retriever import retrieve

MODEL = os.getenv("GENERATIVE_MODEL")
MAX_TOKEN = int(os.getenv("MAX_TOKENS", default=1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", default=0.2))
QDRANT_HOST = os.getenv("QDRANT_HOST", default="localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", default=6333))

client = groq.Groq(api_key=os.getenv("API_KEY"))


def build_context(chunks: list[dict]) -> str:
    logging.debug(f"Building context from {len(chunks)} chunks")
    context = "\n\n---\n\n".join(
        [
            f"Source: {chunk["source_file"]} - {chunk["header_path"]} \n {chunk["content"]}"
            for chunk in chunks
        ]
    )

    logging.debug(f"Context built successfully. Total size: {len(context)} characters")
    return context


def build_prompt(
        question: str,
        context: str
) -> tuple[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]:
    system_prompt = """
    You are a helpful assitant that asnwers questions based on the user's personal Obsidian notes.
    You will only use information present in the provided context.
    If the answer is not in the context, you should say so clearly rather than guessing.
    If you are unable to find the answer from the provided context, you should respond that you don't know,
    instead of giving guessing and providing assumption based answer.
    
    Also, answer the question in a conversational style, no need to explicitly give the content from the context or notes file directly.
    Cite the source file used for refering to specific information but just provide the source names at the end or bottom.
    The response should be like an answer to a question or a explanation to a query, not like a copy paste of source text.
    """

    user_message = f"""
    Context:
    {context}
    
    Question: {question}
    """
    system_prompt = ChatCompletionSystemMessageParam(role="system", content=system_prompt)
    user_message = ChatCompletionUserMessageParam(role="user", content=user_message)
    return system_prompt, user_message

def generate(question: str, chunks: list[dict]):
    logging.debug(f"Generating response for question: {question}")
    logging.debug(f"Number of chunks received: {len(chunks)}")
    context = build_context(chunks)
    logging.debug(f"Context length: {len(context)} characters")
    if not context:
        logging.error("I couldn't find any relevant information in your notes for this question")
        return None

    (system_prompt, user_message) = build_prompt(question, context)

    if not MODEL:
        logging.error("MODEL not provided")
        return None

    logging.debug(f"Calling Groq API with model: {MODEL}, max_tokens: {MAX_TOKEN}, temperature: {TEMPERATURE}")

    response = client.chat.completions.create(
        model=MODEL,
        messages= [system_prompt, user_message],
        max_tokens=MAX_TOKEN,
        temperature=TEMPERATURE,
        stream=True
    )

    complete_response = ""
    finish_reason = ""
    for chunk in response:
        choice = chunk.choices[0]
        token = choice.delta.content
        if token:
            print(token, end="", flush=True)
            complete_response += token
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
            logging.debug(f"Stream finished with reason: {finish_reason}")
    print()

    if finish_reason == "length":
        logging.error("\n[Warning: response was cut off — consider increasing MAX_TOKENS]")

    logging.debug(f"Response generation complete. Length: {len(complete_response)} characters, Finish reason: {finish_reason}")
    return complete_response


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.debug("Starting generator module...")
    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    question = "What is large cap fund"
    # question = "What is the capital of france"
    # question = "What is Mortgage backed security and Collateralized debt obligation"
    chunks = retrieve(question, q_client)
    generate(question, chunks)

    # fake_chunks = [
    #     {
    #         "content": "# 1. Equity Funds\n\nLarge Cap Funds invest in top 100 companies by market cap. Minimum 80% in equity instruments.",
    #         "source_file": "1. Equity Funds.md",
    #         "header_path": "1. Equity Funds",
    #         "folder": "Investment Notes",
    #         "score": 0.52
    #     }
    # ]
    #
    # # something completely unrelated to the context provided
    # question = "What is the best way to learn machine learning?"
    #
    # generate(question, fake_chunks)

