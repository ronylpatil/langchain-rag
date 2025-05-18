# ============== Note: Not in use ==============

import json
from api.logger import infologger
from api.rabbit_utils import connect, publish
from sentence_transformers import CrossEncoder

infologger.info("*** Executing: rank_service.py ***")

model = None


def callback(ch, method, properties, body):
    message = json.loads(body)  
    
    if message.get("stage") != "rank":
        publish("rag_queue", message)
        return

    infologger.info(f"Ranking service received the data.")

    pairs = [(message["user_query"], chunk) for chunk in message["search_results"]]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        infologger.error(f"Failed to rank chunks.")
        raise
    else:
        infologger.info(f"Chunks ranked successfully.")
        ranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, message["search_results"]), reverse=True)
        ]
        
        message["ranked_results"] = ranked_chunks
        message["stage"] = "llm"
        publish("rag_queue", message)


if __name__ == "__main__":

    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        infologger.error("Failed to load CrossEncoder model.")
        raise
    else:
        infologger.info("CrossEncoder model loaded successfully.")

        # Connect to RabbitMQ and start consuming messages
        connection = connect()
        channel = connection.channel()
        channel.queue_declare(queue="rag_queue")
        infologger.info("rag_queue created/declared...")

        channel.basic_consume(
            queue="rag_queue", on_message_callback=callback, auto_ack=True
        )
        infologger.info("Ranking service waiting for message...")
        channel.start_consuming()
