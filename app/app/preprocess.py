import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


# Load the CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()


def get_image_text_similarity(image_path: str, text: str) -> float:
    """
    Computes the cosine similarity between an image and a text description using CLIP.

    Args:
        image_path (str): Path to the input image.
        text (str): Text description to compare with the image.

    Returns:
        float: Cosine similarity score between the image and the text.
    """
    # Preprocess the image and text
    inputs = processor(
        text=[text], 
        images=Image.open(image_path), 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # Extract features from the image and text
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )
    
    # Normalize features to unit vectors
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Debug: Print features
    print("Image Features Shape:", image_features.shape)
    print("Text Features Shape:", text_features.shape)
    print("Image Features:", image_features)
    print("Text Features:", text_features)

    # Compute cosine similarity
    similarity = torch.matmul(image_features, text_features.T).item()
    
    return similarity