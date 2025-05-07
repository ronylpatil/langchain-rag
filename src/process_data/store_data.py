import yaml, os, pathlib
from dotenv import load_dotenv
from src.logger import infologger
from pymilvus import MilvusClient, DataType
from langchain_openai import OpenAIEmbeddings


infologger.info("*** Executing: store_data.py ***")


def connect_milvus(uri: str, token: str) -> MilvusClient:  # STEP: 1
    """
    Connect to Milvus server.

    Args:
        uri (str): Milvus server URI.
        token (str): Milvus token for authentication.

    Returns:
        MilvusClient: MilvusClient Class object.
    """
    try:
        client = MilvusClient(uri=uri, token=token)
        infologger.info("Connected to Milvus successfully.")
    except Exception as e:
        infologger.error(f"Failed to connect to Milvus: {e}")
    else:
        return client


def create_collection(collection_name: str, client: MilvusClient) -> tuple:  # STEP: 2
    # Create schema
    schema = client.create_schema(auto_id=False)
    schema.add_field(
        field_name="doc_id",
        datatype=DataType.INT64,
        is_primary=True,
        description="document id",
    )
    schema.add_field(
        field_name="year",
        datatype=DataType.INT64,
        is_primary=False,
        description="document year",
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1536,
        is_primary=False,
        description="vector",
    )
    schema.add_field(
        field_name="page_content",
        datatype=DataType.VARCHAR,
        max_length=2048,
        is_primary=False,
        description="document content",
    )

    # Delete the collection if it already exists
    if client.has_collection(collection_name):
        infologger.info("Collection already exists. Dropping the collection...")
        try:
            client.drop_collection(collection_name)
        except Exception as e:
            infologger.error(f"Failed to drop the Collection. {e}")
        else:
            # Create new collection
            try:
                client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                )
            except Exception as e:
                infologger.error(f"Failed to create new Collection. {e}")
            else:
                infologger.info("New collection created successfully.")
                return client, schema


def make_index_col(
    field_name: str, collection_name: str, client: MilvusClient
) -> None:  # STEP: 3
    # make vector column as index column
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name=field_name,
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index",
        params={"nlist": 64},
    )

    try:
        client.create_index(
            collection_name=collection_name,
            index_params=index_params,
            sync=False,  # Whether to wait for index creation to complete before returning. Defaults to True.
        )
    except Exception as e:
        infologger.info(f"Failed to create index. {e}")
    else:
        infologger.info("Index created successfully!")
        # Load the collection into memory
        client.load_collection(collection_name="rag_docs")


def load_embedding_model(embedding_model_name: str) -> OpenAIEmbeddings:  # STEP: 4
    try:
        embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        infologger.info("Embedding model loaded successfully.")
    except Exception as e:
        infologger.error(f"Failed to load embedding model: {e}")
    else:
        return embedding_model


def text_to_vector(
    chunks: list, embedding_model_obj: OpenAIEmbeddings
) -> list:  # STEP: 5
    # Create index for the collection
    try:
        vectors = embedding_model_obj.embed_documents(
            list(filter(None, list(map(lambda x: x.page_content, chunks))))
        )
        infologger.info("Embeddings created successfully.")
    except Exception as e:
        infologger.error(f"Failed to convert text to vectors. {e}")
    else:
        return vectors


def prepare_data(chunks: list, embeddings: list) -> list:  # STEP: 6
    ids = []
    vectors = []
    years = []
    page_content = []

    for i in range(len(chunks)):
        ids.append(i)
        vectors.append(embeddings[i])
        years.append(int(chunks[i].metadata["year"]))
        page_content.append(chunks[i].page_content)

    data = [
        {"doc_id": id_val, "vector": vec, "year": yr, "page_content": pc}
        for id_val, vec, yr, pc in zip(ids, vectors, years, page_content)
    ]
    infologger.info("Data prepared successfully.")

    return data


def store_embeddings(
    data: list, collection_name: str, client: MilvusClient
) -> None:  # STEP: 7
    try:
        client.insert(collection_name=collection_name, data=data)
        infologger.info("Embeddings loaded into vector db successfully!")
    except Exception as e:
        infologger.error(f"Failed to load embeddings into vector DB. {e}")


if __name__ == "__main__":

    load_dotenv(override=True)
    uri = os.getenv("uri")
    token = os.getenv("token")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))
    
    # Connect to Milvus Client
    client_obj = connect_milvus(uri, token)
    # Create collection
    client_obj, schema = create_collection(
        params["store_data"]["collection_name"], client_obj
    )
    # Make index column to speed-up the retrieval
    make_index_col(
        params["store_data"]["field_name"],
        params["store_data"]["collection_name"],
        client_obj,
    )
    # Load embedding model
    embedding_model_obj = load_embedding_model(
        params["store_data"]["embedding_model"]
    )

    from src.process_data.chunk_data import load_and_chunk_documents
    # Extract data and split into chunks
    chunks = load_and_chunk_documents(home_dir, params)
    # Convert chunks to vectors
    vectors = text_to_vector(chunks, embedding_model_obj)
    # Process data
    processed_data = prepare_data(chunks, vectors)
    # Store embeddings in vectore store
    store_embeddings(
        data=processed_data,
        collection_name=params["store_data"]["collection_name"],
        client=client_obj,
    )

    infologger.info("*** Completed: store_data.py ***")
