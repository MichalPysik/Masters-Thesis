from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from minio import Minio
from minio.error import S3Error
import os
import ffmpeg
import datetime
from typing import Tuple, Dict
import logging


# Milvus vector database parameters and connection
COLLECTION_NAME = "frame_embedding_collection"

db_host = os.getenv("MILVUS_HOST", "localhost")
db_port = os.getenv("MILVUS_PORT", "19530")
connections.connect(host=db_host, port=db_port)


# Minio storage parameters and connection
BUCKET_NAME = "videos"

minio_host = os.getenv("MINIO_HOST", "localhost")
minio_port = os.getenv("MINIO_PORT", "9000")
minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
minio_client = Minio(
    minio_host + ":" + minio_port,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)


# Map embedding models to corresponding embedding dimensions
EMBEDDING_DIMENSIONS = {
    "CLIP": 768,
    "SigLIP": 1152,
    "ALIGN": 640,
    "BLIP": 256
}

def create_collection(name) -> Collection:
    """
    Creates a new collection in Milvus database with the specified name,
    based on the environment variables (EMBEDDING_MODEL).

    Args:
        name (str): The name of the collection.

    Returns:
        Collection: The created collection object.
    """
    # Define the schema for the collection (fields: id, video_name, timestamp, embedding)
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)  # Primary key
    video_name_field = FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=255)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.FLOAT)

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if embedding_model not in EMBEDDING_DIMENSIONS.keys():
        raise ValueError(f"Invalid embedding model: {embedding_model}")
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS[embedding_model])
    
    schema = CollectionSchema(
        fields=[id_field, video_name_field, timestamp_field, embedding_field],
        description="Embeddings of sampled video frames.",
    )
    collection = Collection(name=name, schema=schema)

    # Create an index for the embedding field
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    # Create the index for the embedding field
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection


def check_bucket_object_exists(object_name: str) -> bool:
    """
    Checks if an object exists in the Minio bucket.

    Args:
        object_name (str): The name of the object to check.

    Returns:
        bool: True if the object exists, False otherwise.
    """
    try:
        minio_client.stat_object(BUCKET_NAME, object_name)
        return True
    except S3Error as err:
        if err.code == "NoSuchKey":
            return False
        raise err


# Returns list of videos in both Minio and Milvus separately
def list_all_data() -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Lists all video data in the Minio bucket and the Milvus database collection.

    Returns:
        Tuple[Dict[str, float], Dict[str, int]]: A tuple containing two dictionaries:
            1. Contains video names from Minio as keys and their corresponding lengths in seconds as values.
            2. Contains unique video names from Milvus as keys and the number of sampled frames as values.
    """
    # Get all video names and lengths from the Minio bucket
    minio_bucket_data = {}
    objects = minio_client.list_objects(BUCKET_NAME)
    minio_video_names = [obj.object_name for obj in objects]

    for video_name in minio_video_names:
        # Generate a presigned URL for the video valid for 1 minute
        url = minio_client.presigned_get_object(BUCKET_NAME, video_name, expires=datetime.timedelta(minutes=1))

         # Probe the video to obtain metadata (including duration)
        try:
            probe = ffmpeg.probe(url)
            video_duration = float(probe['format']['duration'])
            minio_bucket_data[video_name] = video_duration
        except Exception as e:
            minio_bucket_data[video_name] = None
            raise RuntimeError(f"Error retrieving video metadata: {e}")
        
    # Get all video names from Milvus, including the corresponding number of sampled frames
    milvus_collection_data = {}
    collection.load()
    # Query all entities with a non-empty video_name field
    results = collection.query(expr="video_name != ''", output_fields=["video_name"])
    for entity in results:
        video_name = entity.get("video_name")
        if video_name not in milvus_collection_data.keys():
            milvus_collection_data[video_name] = 0
        milvus_collection_data[video_name] += 1

    return minio_bucket_data, milvus_collection_data


def delete_video(video_name: str):
    """
    Deletes a video file from the Minio bucket, along with all related data in the Milvus database collection.

    Args:
        video_name (str): The name of the video file to delete.
    """
    # First delete all corresponding entries from milvus
    collection.load()
    expr = f"video_name == '{video_name}'"
    milvus_delete_response = collection.delete(expr=expr)
    collection.flush()
    if hasattr(milvus_delete_response, "num_deleted"):
        logging.info(f"Deleted {milvus_delete_response.num_deleted} entries for video '{video_name}' from the Milvus database collection.")

    # Then try to delete the video from the bucket
    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file not found in the Minio bucket: {video_name}")
    minio_client.remove_object(BUCKET_NAME, video_name)
    logging.info(f"Deleted video '{video_name}' from Minio bucket '{BUCKET_NAME}'.")


def delete_all_data():
    """
    Deletes all data in the Milvus database collection and in the Minio bucket.
    """
    global collection
    collection.drop()
    collection = create_collection(COLLECTION_NAME)
    logging.info(f"Deleted and created new Milvus database collection '{COLLECTION_NAME}'.")

    objects = minio_client.list_objects(BUCKET_NAME)
    video_names = [obj.object_name for obj in objects]
    for video_name in video_names:
        minio_client.remove_object(BUCKET_NAME, video_name)
    logging.info(f"Deleted all {len(video_names)} videos from Minio bucket '{BUCKET_NAME}': {video_names}")



# Create collection in Milvus if it does not exist
if utility.has_collection(COLLECTION_NAME):
    logging.info(f"Milvus database collection '{COLLECTION_NAME}' already exists.")
    collection = Collection(name=COLLECTION_NAME)
else:
    collection = create_collection(COLLECTION_NAME)
    logging.info(f"Created new Milvus database collection '{COLLECTION_NAME}'.")


# Create bucket in Minio if it does not exist
if minio_client.bucket_exists(BUCKET_NAME):
    logging.info(f"Minio bucket '{BUCKET_NAME}' already exists.")
else:
    minio_client.make_bucket(BUCKET_NAME)
    logging.info(f"Created new Minio bucket '{BUCKET_NAME}'.")