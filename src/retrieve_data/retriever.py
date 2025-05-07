# rank chunks by relevance -- EXPLORE
import re, os, yaml, pathlib
from dotenv import load_dotenv
from src.logger import infologger
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
from pymilvus import MilvusClient, SearchResult
from src.process_data.store_data import connect_milvus, load_embedding_model

infologger.info("*** Executing: retriever.py ***")


def search_relevant_chunks(
    query: str,
    embedding_model: OpenAIEmbeddings,
    collection_name: str,
    client: MilvusClient,
    relevant_chunks: int,
) -> SearchResult:
    """
    Search for relevant chunks in the vector database based on the user query.

    Args:
        query (str): The user query to search in vector database.
        embedding_model (OpenAIEmbeddings): OpenAIEmbeddings object.
        collection_name (str): Collection name.
        client (MilvusClient): MilvusClient object.
        relevant_chunk (int): Number of relevant chunks you want to extract.

    Returns:
        SearchResult: Relevant chunks of type MilvusClient.SearchResult.
    """
    # convert query into embedding
    try:
        query_vector = embedding_model.embed_query(query)
        infologger.info("User Query vectorized successfully.")
    except Exception as e:
        infologger.error("Failed to vectorized user query.")
    else:
        # extract year from query for better retrieval
        year_pattern = r"\b20\d{2}\b"
        year = re.findall(year_pattern, query)
        year = list(map(int, year))

        # Enhancement: handle current year, last year, last n years, last fiscal year
        result = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=relevant_chunks,  # number of top results
            output_fields=["year", "page_content"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 20}},
            filter=f"year in {year}" if year else None,
        )
        infologger.info("Relevant chunks retrieved successfully.")

        return result


def rank_chunks_by_relevance(chunks: SearchResult, query: str) -> tuple:

    try:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        infologger.info("CrossEncoder model loaded.")
        pairs = [(query, chunk.page_content) for chunk in chunks[0]]
        infologger.info("(Query-Chunk) pairs created.")
        scores = model.predict(pairs)
        infologger.info("Scores calculated succesfully.")
    except Exception as e:
        infologger.error(f"Failed to perform ranking. {e}")
    else:
        # Reorder based on scores
        reranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, chunks[0]), reverse=True)
        ]
        infologger.info("Ranking performed succesfully.")

    return scores, reranked_chunks


if __name__ == "__main__":
    load_dotenv(override=True)
    uri = os.getenv("uri")
    token = os.getenv("token")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))

    user_query = input("Enter query: ")

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
    for score, chunk in zip(scores, ranked_chunks):
        print(f"Score: {score:.2f} Content: {chunk.page_content[:100]}...")

    # import textwrap

    # for i in chunks[0]:
    #     print(textwrap.fill(i.page_content, 80))
    #     print("Source Year: ", i.year)
    #     print("=" * 75)

    infologger.info("*** Completed: retriever.py ***")
