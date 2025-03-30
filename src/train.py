import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

# Set fixed seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Paths
PREPROCESSED_DIR = "data/preprocessed"
TRAIN_DIR = os.path.join(PREPROCESSED_DIR, "train")
IMAGES_DIR = TRAIN_DIR
CAPTIONS_FILE = os.path.join(TRAIN_DIR, "captions.txt")
CONFIGS_DIR = "configs"
RESULTS_DIR = "results"
MODEL_SAVE_PATH = os.path.join(CONFIGS_DIR, "clip_flickr8k.pt")
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LR = 5e-6
MAX_CAPTIONS_PER_IMAGE = 1

# Processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, max_captions_per_image=1):
        self.image_dir = image_dir
        self.data = []

        rows = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            next(f)
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
                image_path = os.path.join(self.image_dir, image)
                if not os.path.isfile(image_path):
                    print(f"Warning: image file not found - {image_path}")
                    continue
                rows.append({'image': image, 'caption': caption})
        df = pd.DataFrame(rows)

        if max_captions_per_image is not None:
            df = df.groupby('image').head(max_captions_per_image).reset_index(drop=True)

        self.data = df.to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.image_dir, sample['image'])
        image = Image.open(image_path).convert('RGB')
        caption = sample['caption']
        return {'image': image, 'caption': caption, 'meta_image_id': sample['image']}

def collate_fn(batch):
    try:
        images = [item['image'] for item in batch]
        captions = [item['caption'] for item in batch]
        meta_image_ids = [item['meta_image_id'] for item in batch]

        processed = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        if processed['input_ids'].shape[0] != processed['pixel_values'].shape[0]:
            raise ValueError("Mismatched text and image batch sizes after processing")

        processed['meta_image_id'] = meta_image_ids
        processed['meta_caption'] = captions
        return processed
    except Exception as e:
        print(f"Skip for collate function failed: {e}")
        return None

def train_clip():
    print("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    print("Preparing dataset...")
    dataset = FlickrDataset(IMAGES_DIR, CAPTIONS_FILE, max_captions_per_image=MAX_CAPTIONS_PER_IMAGE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(dataloader))

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            if batch is None:
                continue
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                if inputs['input_ids'].shape[0] != inputs['pixel_values'].shape[0]:
                    print("Skip for mismatched batch size between text and image inputs")
                    continue

                outputs = model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds

                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_embeds @ text_embeds.T
                logits_per_text = logits_per_image.T

                labels = torch.arange(logits_per_image.size(0), device=device)
                loss = (
                    torch.nn.functional.cross_entropy(logits_per_image, labels) +
                    torch.nn.functional.cross_entropy(logits_per_text, labels)
                ) / 2

                if not torch.isfinite(loss):
                    print("Skip for invalid loss")
                    continue

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Skip for exception in batch: {e}")
                continue

        if num_batches == 0:
            print(f"Epoch {epoch+1} - No valid batches")
        else:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def main():
    try:
        train_clip()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        exit(1)

if __name__ == "__main__":
    main()