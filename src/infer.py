import os
import torch
import pandas as pd
import random
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Paths
MODEL_WEIGHTS_PATH = "configs/clip_flickr8k.pt"
TEST_DIR = "data/preprocessed/test"
IMAGES_DIR = TEST_DIR
CAPTIONS_FILE = os.path.join(TEST_DIR, "captions.txt")
RESULTS_DIR = "results"
INFERENCE_FILE = os.path.join(RESULTS_DIR, "inference_results.txt")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_caption_mapping():
    """Loads the mapping of images to captions."""
    df = pd.read_csv(CAPTIONS_FILE)
    mapping = []
    for _, row in df.iterrows():
        image_path = os.path.join(IMAGES_DIR, row['image'])
        if os.path.exists(image_path):
            mapping.append((image_path, row['caption']))
        else:
            print(f"Skip for image not found: {image_path}")
    print(f"Loaded {len(mapping)} valid image-caption pairs.")
    return mapping

def infer_sampled(mapping, sample_size=50, top_k=5):
    print("Loading the CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(mapping) == 0:
        print("No valid test samples found.")
        return

    sampled = random.sample(mapping, min(sample_size, len(mapping)))
    all_captions = [cap for _, cap in mapping]

    with open(INFERENCE_FILE, "w", encoding='utf-8') as f:
        for idx, (img_path, true_caption) in enumerate(sampled):
            image = Image.open(img_path).convert("RGB")

            inputs = processor(
                text=all_captions,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                sims = image_embeds @ text_embeds.T
                topk_indices = sims[0].topk(top_k).indices.cpu().numpy()

            f.write(f"Image: {os.path.basename(img_path)}\n")
            f.write(f"True Caption: {true_caption}\n")
            f.write(f"Top-{top_k} Predicted Captions:\n")
            for i in topk_indices:
                f.write(f"  - {all_captions[i]}\n")
            f.write("\n")

            print(f"[{idx+1}/{sample_size}] Processed {os.path.basename(img_path)}")

    print(f"Inference completed. Results saved to {INFERENCE_FILE}")

def main():
    try:
        mapping = load_caption_mapping()
        infer_sampled(mapping, sample_size=50)
    except Exception as e:
        print(f"Error during inference: {e}")
        exit(1)

if __name__ == "__main__":
    main()