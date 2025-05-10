import json
import yaml
import pathlib
from logger import infologger
from rabbit_utils import connect, publish
from langchain_openai import OpenAIEmbeddings

infologger.info("*** Executing: vectorizer_service.py ***")

embedding_model_obj = None


def callback(ch, method, properties, body) -> None:
    data = json.loads(body)  # body: {"text": input.text}
    infologger.info(f"Data received at vectorizer_queue.")
    # infologger.info(f"Data: {data}")

    try:
        query_vector = embedding_model.embed_query(data["text"])
    except Exception as e:
        infologger.error(f"Failed to vectorize user query.")
        raise
    else:
        infologger.info(f"User query vectorized successfully.")

        # Forward user query + vector to search_queue
        publish("search_queue", {"query": data["text"], "vector": query_vector})


if __name__ == "__main__":

    home_dir = pathlib.Path(__file__).parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    try:
        embedding_model = OpenAIEmbeddings(
            model=params["store_data"]["embedding_model"]
        )
    except Exception as e:
        infologger.error(f"Failed to load embedding model: {e}")
        raise
    else:
        # Connect to RabbitMQ and start consuming messages
        connection = connect()
        channel = connection.channel()
        channel.queue_declare(queue="vectorizer_queue")
        infologger.info("vectorizer_queue created/declared...")

        channel.basic_consume(
            queue="vectorizer_queue", on_message_callback=callback, auto_ack=True
        )
        infologger.info("vectorizer_queue waiting for text...")
        channel.start_consuming()
