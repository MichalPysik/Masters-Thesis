import torch

from app import collection, processor, device, model


# Function to search for the most relevant frames given a text input
def search_similar_frames(text: str, top_k: int = 5):
    # Load the collection if it's not already loaded
    collection.load()

    # Tokenize the input text and generate embedding
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Normalize the text embedding
    text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Convert text feature to numpy array
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

    # Debugging: Print out the raw search results
    print("Search results:", search_results)

    return search_results


