# load data from S3
import yaml
import boto3
import pathlib
from dotenv import load_dotenv
from src.logger import infologger


infologger.info("*** Executing: load_data.py ***")

load_dotenv()


def load_data_from_s3(params: dict) -> None:
    """
    Load documents from S3 bucket to local directory
    """
    # Connect to S3
    try:
        s3 = boto3.client("s3")
        infologger.info("Connected to S3 successfully")
    except Exception as e:
        infologger.error(f"Error connecting to S3: {e}")

    # List all docs under the given prefix
    try:
        response = s3.list_objects_v2(
            Bucket=params["upload_to_s3"]["s3_bucket"],
            Prefix=params["upload_to_s3"]["s3_path"],
        )
    except Exception as e:
        infologger.error(f"Error listing objects in S3: {e}")

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pdf"):
            try:
                s3.download_file(
                    params["upload_to_s3"]["s3_bucket"],
                    key,
                    f"{home_dir}/data/raw/{key.split('/')[-1]}",
                )
                infologger.info(
                    f"Downloaded {key} to {home_dir}/data/raw/{key.split('/')[-1]}"
                )
            except Exception as e:
                infologger.error(f"Error downloading {key}: {e}")


if __name__ == "__main__":
    # load parameters from params.yaml
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml", encoding="utf-8"))
    load_data_from_s3(params)
    infologger.info("*** Completed: load_data.py ***")
