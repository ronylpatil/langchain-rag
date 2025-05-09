import json
import yaml
import pathlib
from src.logger import infologger
from rabbit_utils import connect, publish
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

infologger.info("*** Executing: rank_service.py ***")

model = None

def callback(ch, method, properties, body):
    data = json.loads(body)  # {"query": data["query"], "ranked_chunks": ranked_chunks}
    infologger.info(f"Data received at llm_queue.")
    infologger.info(f"Data: {data}")
    
    context = "\n".join([i.page_content for i in data["ranked_chunks"]])
    infologger.info("Final context prepared.")

    try:
        curr_dir = pathlib.Path(__file__).parent.as_posix()
        with open(f"{curr_dir}/prompt.j2", "r") as file:
            raw_template = file.read()
    except Exception as e:
        infologger.error(f"Failed to load prompt.")
        raise
    else:
        infologger.info("Prompt file loaded successfully.")
        prompt_temp = PromptTemplate(
            input_variables=["context", "question"], template=raw_template
        )

        prompt = prompt_temp.invoke({"context": f"{context}", "question": f"{data['query']}"})
        
        try:
            response = model.invoke(prompt)
        except Exception as e:
            infologger.error(f"Failed to invoke LLM model. {e}")
        else:
            infologger.info("LLM resonse generated successfully.")
            print(f"Final response: {response}")

    
if __name__ == "__main__":

    home_dir = pathlib.Path(__file__).parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    try:
        model = ChatOpenAI(model=params["llm_response"]["model"])
    except Exception as e:
        infologger.error(f"Failed to load {params["llm_response"]["model"]} model: {e}")
        raise
    else:
        infologger.info(
            f"Successfully loaded {params["llm_response"]["model"]} model with {params["llm_response"]["temprature"]} temprature."
        )
    
    connection = connect()
    channel = connection.channel()
    channel.queue_declare(queue="llm_queue")
    infologger.info("llm_queue created/declared...")
    
    channel.basic_consume(
        queue="llm_queue", on_message_callback=callback, auto_ack=True
    )
    infologger.info("vectorizer_queue waiting for text...")
    channel.start_consuming()
