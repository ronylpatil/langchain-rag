import json
import yaml
import pathlib
from api.logger import infologger
from api.rabbit_utils import connect, publish
from langchain_openai import OpenAIEmbeddings

infologger.info("*** Executing: vectorizer_service.py ***")

embedding_model_obj = None


def callback(ch, method, properties, body) -> None:
    message = json.loads(body)
    if message.get("stage") != "vectorizer":
        publish("rag_queue", message)
        return
        
    infologger.info(f"Vectorizer service received the data.")
    infologger.info(f"User Query: {message['user_query']}")
    try:
        query_vector = embedding_model.embed_query(message["user_query"])
        message["query_vector"] = query_vector
    except Exception as e:
        infologger.error(f"Failed to vectorize user query.")
        raise
    else:
        infologger.info(f"User query vectorized successfully.")
        message["stage"] = "search"

        # Forward user query + vector to search_queue
        publish("rag_queue", message)
        

if __name__ == "__main__":

    home_dir = pathlib.Path(__file__).parent.parent.parent.as_posix()
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
        channel.queue_declare(queue="rag_queue")
        infologger.info("rag_queue created/declared...")

        channel.basic_consume(
            queue="rag_queue", on_message_callback=callback, auto_ack=True
        )
        infologger.info("Vectorizer service waiting for message...")
        channel.start_consuming()
