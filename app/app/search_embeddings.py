import torch
import os

from app import collection, device, emb_model, emb_processor


# Function to search for the most relevant frames given a text input
def search_similar_frames(text: str, top_k: int = 5):
    # Load the collection if it's not already loaded
    collection.load()

    # Process the input text and generate embedding
    if os.getenv("EMBEDDING_MODEL") == "Blip":
        inputs = emb_processor[1]["eval"](text)
        sample = {"image": None, "text_input": [inputs]}
        text_features = emb_model.extract_features(sample, mode="text")
        # Project from 768 to 256 dimensions (includes normalization)
        text_features = text_features.text_embeds_proj[:, 0, :]
    else:
        inputs = emb_processor(text=text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = emb_model.get_text_features(**inputs)
        # Normalize the embedding
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Convert the text embedding to 1-D numpy array
    query_embedding = text_features.cpu().numpy().flatten()

    # Search in Milvus
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "top_k": top_k},
        consistency_level="Strong",
        output_fields=["video_name", "timestamp"],
        limit=top_k,
    )

    # Debug: Print out the raw search results
    print("Search results:", search_results)

    return search_results


