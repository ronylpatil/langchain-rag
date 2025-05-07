# chunk text into smaller pieces
import re
import yaml
import pathlib
from src.logger import infologger
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

infologger.info("*** Executing: chunk_data.py ***")


def load_and_chunk_documents(home_dir: str, params: dict) -> list:
    """
    Load data from local directory and splitted into chunks.

    Args:
        home_dir (str): Home Directory path.
        params (dict): Parameters from params.yaml

    Returns:
        Return list of chunks.
    """
    # DirectoryLoader initialization
    try:
        dir_loader = DirectoryLoader(
            f"{home_dir}/{params['s3_loader']['local_data']}",
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
        infologger.info("DirectoryLoader initialized successfully")
    except Exception as e:
        infologger.error(f"Error in DirectoryLoader: {e}")

    # Extract year from the file name
    docs_with_year = []
    for doc in dir_loader.lazy_load():
        # adding budget year to metadata
        doc.metadata["year"] = doc.metadata["source"].split("\\")[-1].split("_")[0][-4:]

        # remove page number
        doc.page_content = re.sub(r"^\d{1,2}\s", "", doc.page_content)
        # remove extra spaces
        doc.page_content = re.sub(r"\s+", " ", doc.page_content)

        if doc.page_content.strip():
            docs_with_year.append(doc)

    infologger.info(
        f"Performed data cleaning and added year as metadata for {len(docs_with_year)} documents"
    )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_data"]["chunk_size"],
            chunk_overlap=params["chunk_data"]["chunk_overlap"],
        )
        chunks = splitter.split_documents(docs_with_year)
        infologger.info(f"documents chunked into {len(chunks)} pieces")
    except Exception as e:
        infologger.error(f"Error in RecursiveCharacterTextSplitter: {e}")

    return chunks


if __name__ == "__main__":

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))
    chunks = load_and_chunk_documents(home_dir, params)
    print(chunks[5])
    infologger.info("*** Completed: chunk_data.py ***")
