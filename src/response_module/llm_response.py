import os, yaml, pathlib
from dotenv import load_dotenv
from src.logger import infologger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.retrieve_data.retriever import search_relevant_chunks, rank_chunks_by_relevance
from src.process_data.store_data import connect_milvus, load_embedding_model


infologger.info("*** Executing: llm_response.py ***")


def load_model(model: str, temprature: float) -> ChatOpenAI:
    try:
        llm = ChatOpenAI(model=model)
    except Exception as e:
        infologger.error(f"Failed to load {model} model: {e}")
        raise
    else:
        infologger.info(
            f"Successfully loaded {model} model with {temprature} temprature."
        )
        return llm


def process_text(chunks: list) -> str:
    try:
        final_context = "\n".join([i.page_content for i in chunks])
        infologger.info("Chunks processed successfully.")
        return final_context
    except Exception as e:
        infologger.error(f"Failed to process chunks. {e}")


def response(
    query: str,
    context: str,
    model: ChatOpenAI,
    prompt_file_path: str,
) -> str:
    # prompt ko LLM ko bhejo aur response lo
    try:
        with open(prompt_file_path, "r") as file:
            raw_template = file.read()
    except Exception as e:
        infologger.error(f"Failed to load prompt. {e}")
    else:
        infologger.info("Prompt file loaded successfully.")
        prompt = PromptTemplate(
            input_variables=["context", "question"], template=raw_template
        )

        final_prompt = prompt.invoke({"context": f"{context}", "question": f"{query}"})
        
        try:
            final_response = model.invoke(final_prompt)
        except Exception as e:
            infologger.error(f"Failed to invoke LLM model. {e}")
        else:
            infologger.info("LLM resonse generated successfully.")
            return final_response


if __name__ == "__main__":
    load_dotenv(override=True)
    uri = os.getenv("uri")
    token = os.getenv("token")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))
    llm = load_model(
        params["llm_response"]["model"], params["llm_response"]["temprature"]
    )

    user_query = "what is goverment plan for farmers in 2025?"

    client_obj = connect_milvus(uri, token)
    embedding_model_obj = load_embedding_model(
        params["store_data"]["embedding_model"]
    )

    chunks = search_relevant_chunks(
        query=user_query,
        embedding_model=embedding_model_obj,
        collection_name=params["store_data"]["collection_name"],
        client=client_obj,
        relevant_chunks=params["retriever"]["relevant_chunks"]
    )

    scores, ranked_chunks = rank_chunks_by_relevance(chunks, user_query)
    
    final_context = process_text(ranked_chunks)
    final_response = response(query=user_query, context=final_context, model=llm, prompt_file_path=params["llm_response"]["prompt_file_path"])
    print(f"Response: {final_response.content}")
    
    infologger.info("*** Completed: llm_response.py ***")
