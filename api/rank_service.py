import json
from src.logger import infologger
from rabbit_utils import connect, publish
from sentence_transformers import CrossEncoder

infologger.info("*** Executing: rank_service.py ***")

model = None


def callback(ch, method, properties, body):
    data = json.loads(body)  # {"query": data["query"], "top_chunks": result}
    infologger.info(f"Data received at rank_queue.")
    infologger.info(f"Data: {data}")

    pairs = [(data["query"], chunk) for chunk in data["top_chunks"]]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        infologger.error(f"Failed to rank chunks.")
        raise
    else:
        infologger.info(f"Chunks ranked successfully.")
        ranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, data["top_chunks"]), reverse=True)
        ]

        publish("llm_queue", {"query": data["query"], "ranked_chunks": ranked_chunks})
        infologger.info("Data sent to llm_queue successfully...")


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
        channel.queue_declare(queue="rank_queue")
        infologger.info("rank_queue created/declared...")

        channel.basic_consume(
            queue="rank_queue", on_message_callback=callback, auto_ack=True
        )
        infologger.info("rank_queue waiting for vectors...")
        channel.start_consuming()
