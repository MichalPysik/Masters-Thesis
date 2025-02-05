from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from minio import Minio
from minio.error import S3Error
import os


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
        description="Embeddings of video frames.",
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

# Returns list of videos in both Minio and Milvus, list of minio-only, and list of milvus-only
def show_all_videos():
    pass

def delete_video(video_name: str):
    """
    Deletes a video file from the Minio bucket, along with all related data in the Milvus database.

    Args:
        video_name (str): The name of the video file to delete.
    """
    # First delete all corresponding entries from milvus
    expr = f"video_name == '{video_name}'"
    milvus_delete_response = collection.delete(expr=expr)
    collection.flush()
    if hasattr(milvus_delete_response, "num_deleted"):
        print(f"Deleted {milvus_delete_response.num_deleted} entries from Milvus for video '{video_name}'.")

    # Then try to delete the video from the bucket
    if not check_bucket_object_exists(video_name):
        raise FileNotFoundError(f"Video file not found in the bucket: {video_name}.")
    minio_client.remove_object(BUCKET_NAME, video_name)
    print(f"Deleted video '{video_name}' from bucket '{BUCKET_NAME}'.")


def delete_all_data():
    """
    Deletes all data in the Milvus collection and the Minio bucket.
    """
    global collection
    collection.drop()
    collection = create_collection(COLLECTION_NAME)
    print(f"Deleted and created new collection '{COLLECTION_NAME}'.")

    objects = minio_client.list_objects(BUCKET_NAME)
    num_deleted = 0
    for obj in objects:
        minio_client.remove_object(BUCKET_NAME, obj.object_name)
        num_deleted += 1
    print(f"Deleted all {num_deleted} videos from bucket '{BUCKET_NAME}'.")


def synchronize_data():
    """
    Synchronizes the data in the Milvus collection with the videos in the Minio bucket.
    It deletes any entries in the Milvus collection that do not have a corresponding video in the bucket and vice versa.
    It also creates and fills (according to the bucket videos) a new collection when the embedding model changes.
    """
    pass


# Create collection in Milvus if it does not exist
if utility.has_collection(COLLECTION_NAME):
    print(f"Milvus collection '{COLLECTION_NAME}' already exists.")
    collection = Collection(name=COLLECTION_NAME)
else:
    print(f"Creating new Milvus collection '{COLLECTION_NAME}'.")
    collection = create_collection(COLLECTION_NAME)


# Create bucket in Minio if it does not exist
if minio_client.bucket_exists(BUCKET_NAME):
    print(f"Minio bucket '{BUCKET_NAME}' already exists.")
else:
    print(f"Creating new Minio bucket '{BUCKET_NAME}'.")
    minio_client.make_bucket(BUCKET_NAME)