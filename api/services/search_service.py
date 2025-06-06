# ============== Note: Not in use ==============

import os
import re
import json
from api.logger import infologger
from dotenv import load_dotenv
from pymilvus import MilvusClient
from api.rabbit_utils import connect, publish


infologger.info("*** Executing: search_service.py ***")

client = None


def callback(ch, method, properties, body):
    message = json.loads(body)
    
    if message.get("stage") != "search":
        publish("rag_queue", message)
        return

    infologger.info(f"Search service received the data.")

    # extract year from query for better retrieval
    year_pattern = r"\b20\d{2}\b"
    year = re.findall(year_pattern, message["user_query"])
    year = list(map(int, year))
    infologger.info(f"Extracted year from query: {year}")

    # Enhancement: handle current year, last year, last n years, last fiscal year
    try:
        result = client.search(
            collection_name="rag_docs",
            data=[message["query_vector"]],  # query vector
            limit=5,  # number of top results
            output_fields=["year", "page_content"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 20}},
            filter=f"year in {year}" if year else None,
        )
    except Exception as e:
        infologger.error(f"Failed to search relevant chunks.")
        raise
    else:
        infologger.info("Relevant chunks extracted successfully.")

        # Processing the result to get the top chunks
        top_chunks = []
        for i in result[0]:
            top_chunks.append(i.page_content)

        message["search_results"] = top_chunks
        message["stage"] = "rank"

        # Forward user query + top chunks to rank_queue
        publish("rag_queue", message)


if __name__ == "__main__":

    load_dotenv(override=True)
    uri = os.getenv("uri")
    token = os.getenv("token")

    try:
        client = MilvusClient(uri=uri, token=token)
    except Exception as e:
        infologger.error(f"Failed to initilize MilvusClient.")
        raise
    else:
        infologger.info("MilvusClient intilized successfully.")

        connection = connect()
        channel = connection.channel()
        channel.queue_declare(queue="rag_queue")
        infologger.info("rag_queue created/declared...")

        channel.basic_consume(
            queue="rag_queue", on_message_callback=callback, auto_ack=True
        )
        infologger.info("Search service waiting for message...")
        channel.start_consuming()
