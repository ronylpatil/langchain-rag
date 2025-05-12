import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from api.rabbit_utils import publish  # Reuse your existing RabbitMQ publishing code

app = FastAPI()  # Create the FastAPI app


# Define input structure
class QueryInput(BaseModel):
    text: str


@app.post("/query")
async def send_query(input: QueryInput):
    """
    This endpoint receives user text (query),
    and sends it to the vectorizer_queue to start the pipeline.

    Args:
        input (QueryInput): The input data containing the text to be vectorized.
    """

    query_id = str(uuid.uuid4())  # Generate a unique query ID
    # Message template
    message = {
        "id": query_id,
        "stage": "vectorizer",
        "user_query": input.text,
        "query_vector": None,
        "search_results": None,
        "ranked_results": None,
        "final_answer": None,
    }

    # Send user query to vectorizer_queue
    publish("rag_queue", message)

    return {"info": "query sent to message queue.", "query_id": query_id}


# CMD: uvicorn api.api_service:app --reload
