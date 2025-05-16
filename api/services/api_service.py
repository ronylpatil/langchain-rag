import re
import os
import asyncio
import pathlib
from pydantic import BaseModel
from dotenv import load_dotenv
from api.logger import infologger
from fastapi import FastAPI, HTTPException
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate
from pymilvus import MilvusClient, MilvusException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)
uri = os.getenv("uri")
token = os.getenv("token")


app = FastAPI()  # Create the FastAPI app

try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_model = ChatOpenAI(model="gpt-4.1-mini")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

try:
    client = MilvusClient(uri=uri, token=token)
except MilvusException as e:
    error_message = str(e.args[0]) if e.args else str(e)
    raise HTTPException(
        status_code=500,
        detail=f"\n{'='*20} ERROR {'='*20}\nError: Milvus error, check credentials. \n{error_message}\n{'='*20} ERROR {'='*20}",
    )
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

try:
    with open(
        f"{pathlib.Path(__file__).parent.parent.as_posix()}/prompt.j2", "r"
    ) as file:
        raw_template = file.read()
    infologger.info("Prompt file loaded successfully.")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Prompt file not found.")


# Define input structure
class QueryInput(BaseModel):
    text: str


class InvalidQueryInputError(Exception):
    """Raised when the user query is missing or invalid"""

    pass


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails"""

    pass


# Step 1: Embed user query
def compute_embedding(user_query: QueryInput) -> list:
    try:
        if not hasattr(user_query, "text") or not user_query.text:
            raise InvalidQueryInputError(
                "QueryInput must have a non-empty 'text' field."
            )
        query_vector = embedding_model.embed_query(user_query.text)
        infologger.info(f"User query embedded successfully.")

        return query_vector
    except InvalidQueryInputError as e:
        raise HTTPException(status_code=400, detail="Invalid query input.")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"\n{'='*20} ERROR {'='*20}\nError: Milvus error, check credentials. \n{error_message}\n{'='*20} ERROR {'='*20}",
        )


# Step 2: Search in Milvusdb
def search_top_k(query_vector: list, user_query: QueryInput) -> list:
    year_pattern = r"\b20\d{2}\b"
    year = re.findall(year_pattern, user_query.text)
    year = list(map(int, year))
    if year:
        infologger.info(f"Year mentioned in user query.")

    result = client.search(
        collection_name="rag_docs",
        data=[query_vector],  # query vector
        limit=5,  # number of top results
        output_fields=["year", "page_content"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 20}},
        filter=f"year in {year}" if year else None,
    )
    infologger.info(f"Top 5 chunks retrieved successfully.")

    if not result or not result[0]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    top_chunks = []
    for i in result[0]:
        top_chunks.append(i.page_content)
    return top_chunks


# Step 3: Perform context-based ranking
def rerank(top_chunks: list, user_query: QueryInput) -> list:
    try:
        pairs = [(user_query.text, chunk) for chunk in top_chunks]
        if not pairs:
            raise ValueError("Pairs list is empty.")

        scores = model.predict(pairs)
        ranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, top_chunks), reverse=True)
        ]
        return ranked_chunks
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed. {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"\nError: Unexpected error while ranking the chunks. \n{str(e)}",
        )


# Step 4: Async LLM generation
async def generate_answer(
    context: str, raw_template: str, user_query: QueryInput
) -> str:
    try:
        prompt_temp = PromptTemplate(
            input_variables=["context", "question"], template=raw_template
        )

        prompt = prompt_temp.invoke(
            {"context": f"{context}", "question": f"{user_query.text}"}
        )
        infologger.info("Prompt template invoked successfully.")

        response = llm_model.invoke(prompt)
        return response.content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"\n{'='*20} ERROR {'='*20}\nError: LLM output generation failed. \n{str(e)}\n{'='*20} ERROR {'='*20}",
        )


@app.post("/query")
async def send_query(input: QueryInput) -> None:
    """
    This endpoint receives user querys, processes them through a series of steps including embedding,
    vector search, ranking, and LLM generation, and returns the final answer.

    Args:
        input (QueryInput): The input data containing the text to be vectorized.
        
    Returns:
        dict: A dictionary containing the generated answer.
    """

    loop = asyncio.get_event_loop()

    # Step 1: Embedding
    embedding = await loop.run_in_executor(None, compute_embedding, input)
    infologger.info(f"User query vectorized successfully.")

    # Step 2: Vector Search
    chunks = await loop.run_in_executor(None, search_top_k, embedding, input)
    infologger.info(f"Top 5 chunks retrieved successfully.")

    # Step 3: Reranking
    top_chunks = await loop.run_in_executor(None, rerank, chunks, input)
    infologger.info(f"Top 5 chunks ranked successfully.")

    # Step 4: LLM Generation
    context = "\n".join([i for i in top_chunks])
    infologger.info(f"Almost there...")
    answer = await generate_answer(context, raw_template, input)
    infologger.info("LLM response generated successfully.")
    return {"answer": answer}


# CMD: uvicorn api.services.api_service:app --reload