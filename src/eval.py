import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.metrics import accuracy_score

# Paths
MODEL_WEIGHTS_PATH = "configs/clip_flickr8k.pt"
PREPROCESSED_DIR = "data/preprocessed"
TEST_DIR = os.path.join(PREPROCESSED_DIR, "test")
IMAGES_DIR = TEST_DIR
CAPTIONS_FILE = os.path.join(TEST_DIR, "captions.txt")
RESULTS_DIR = "results"
EVALUATION_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_caption_mapping():
    """Loads the mapping of images to captions."""
    mapping = []
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        next(f)  # Skip header if needed
        for line in f:
            if '#0 ' not in line and ',' not in line:
                continue
            if '#0 ' in line:
                parts = line.strip().split('#0 ', 1)
            else:
                parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            image, caption = parts
            image_path = os.path.join(IMAGES_DIR, image)
            if os.path.exists(image_path):
                mapping.append((image_path, caption))
    return mapping

def evaluate_model():
    """
    Evaluates the fine-tuned CLIP model using batched similarity scoring.
    """
    print("Loading the CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Loading image-caption pairs...")
    mapping = load_caption_mapping()
    image_paths, captions = zip(*mapping)

    print("Preprocessing all inputs...")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

    print("Computing embeddings...")
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    print("Computing similarity matrix...")
    similarity = image_embeds @ text_embeds.T
    preds = similarity.argmax(dim=1)
    labels = torch.arange(len(preds), device=preds.device)
    accuracy = (preds == labels).float().mean().item()

    with open(EVALUATION_FILE, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Total Samples: {len(labels)}\n")
        f.write(f"Top-1 Accuracy: {accuracy:.4f}\n")

    print(f"Evaluation metrics saved to {EVALUATION_FILE}")
    print("Evaluation completed successfully!")

def main():
    try:
        evaluate_model()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()