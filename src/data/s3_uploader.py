# push data from local -> s3
import yaml
import boto3
import pathlib
from dotenv import load_dotenv
from src.logger import infologger

infologger.info("*** Executing: upload_to_s3.py ***")

load_dotenv(override=True)


def push_to_s3(params: dict) -> None:
    """
    Uploads all pdf files from the local directory to S3 bucket.
    """
    # Connect to S3
    try:
        s3 = boto3.client("s3")
        infologger.info("Connected to S3 successfully")
    except Exception as e:
        infologger.error(f"Error connecting to S3: {e}")

    # upload all pdf files
    for file in pathlib.Path(f"{home_dir}/{params['data_path']}").rglob("*"):
        if str(file).split(".")[-1] == "pdf":
            try:
                file_name = str(file).split("\\")[-1]
                s3.upload_file(
                    file, f"{params['s3_bucket']}", f"{params['s3_path']}/{file_name}"
                )
                infologger.info(
                    f"File: {file} uploaded to s3://{params['s3_path']}/{file_name}"
                )
            except Exception as e:
                infologger.error(f"Error uploading File: {file} to S3: {e}")


if __name__ == "__main__":
    # load parameters from params.yaml
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params_loc = home_dir + "/params.yaml"
    params = yaml.safe_load(open(params_loc, encoding="utf-8"))["upload_to_s3"]
    # push data from local -> s3
    push_to_s3(params)
    infologger.info("*** Completed: upload_to_s3.py ***")
