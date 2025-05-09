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
    # Send user query to vectorizer_queue
    publish("vectorizer_queue", {"text": input.text})

    return {"message": "query sent to vectorizer_queue..."}

# CMD: uvicorn api.api_service:app --reload
