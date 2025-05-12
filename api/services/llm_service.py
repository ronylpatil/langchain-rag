import json
import yaml
import pathlib
from api.logger import infologger
from api.rabbit_utils import connect, publish
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

infologger.info("*** Executing: rank_service.py ***")

model = None


def callback(ch, method, properties, body):
    message = json.loads(body)

    if message.get("stage") != "llm":
        publish("rag_queue", message)
        return

    infologger.info(f"LLM service received the data.")

    context = "\n".join([i for i in message["ranked_results"]])
    infologger.info("Final context prepared.")

    try:
        # curr_dir = pathlib.Path(__file__).parent.parent.as_posix()
        with open(
            f"{pathlib.Path(__file__).parent.parent.as_posix()}/prompt.j2", "r"
        ) as file:
            raw_template = file.read()
    except Exception as e:
        infologger.error(f"Failed to load prompt.")
        raise
    else:
        infologger.info("Prompt file loaded successfully.")
        prompt_temp = PromptTemplate(
            input_variables=["context", "question"], template=raw_template
        )

        prompt = prompt_temp.invoke(
            {"context": f"{context}", "question": f"{message['user_query']}"}
        )

        try:
            response = model.invoke(prompt)
        except Exception as e:
            infologger.error(f"Failed to invoke LLM model. {e}")
        else:
            message["final_answer"] = response.content
            message["stage"] = "completed"
            infologger.info("LLM resonse generated successfully.")
            print(f"Query: {message['user_query']}")
            print(f"Final response: {response.content}")

    ch.basic_ack(delivery_tag=method.delivery_tag)

    # NOTE: if any error came in this stage then requeue the message - Add this condition to callback


if __name__ == "__main__":

    home_dir = pathlib.Path(__file__).parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    try:
        model = ChatOpenAI(model=params["llm_response"]["model"])
    except Exception as e:
        infologger.error(f"Failed to load {params['llm_response']['model']} model: {e}")
        raise
    else:
        infologger.info(
            f"Successfully loaded {params['llm_response']['model']} model with {params['llm_response']['temprature']} temprature."
        )

    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue="rag_queue")
    infologger.info("rag_queue created/declared...")

    channel.basic_consume(
        queue="rag_queue", on_message_callback=callback, auto_ack=False
    )
    infologger.info("LLM service waiting for message...")
    channel.start_consuming()
