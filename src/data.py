import os
import shutil
import random
import kagglehub

# Paths
RAW_DATA_DIR = "data/raw"
PREPROCESSED_DATA_DIR = "data/preprocessed"
TRAIN_DIR = os.path.join(PREPROCESSED_DATA_DIR, "train")
TEST_DIR = os.path.join(PREPROCESSED_DATA_DIR, "test")
SPLIT_RATIO = 0.8

def download_dataset():
    """
    Downloads the Flickr8k dataset using kagglehub and splits it into train/test.
    """
    print("Downloading the Flickr8k dataset from KaggleHub...")
    dataset_dir = kagglehub.dataset_download("adityajn105/flickr8k")
    print(f"Dataset downloaded to: {dataset_dir}")

    # Cleanup existing raw directory
    if os.path.exists(RAW_DATA_DIR):
        print(f"Removing existing directory: {RAW_DATA_DIR}")
        shutil.rmtree(RAW_DATA_DIR)
    print(f"Moving dataset directory to {RAW_DATA_DIR}...")
    shutil.move(dataset_dir, RAW_DATA_DIR)

    # Prepare clean preprocessed directory
    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Removing existing directory: {PREPROCESSED_DATA_DIR}")
        shutil.rmtree(PREPROCESSED_DATA_DIR)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Split image files
    image_dir = os.path.join(RAW_DATA_DIR, "Images")
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    random.shuffle(image_files)
    split_idx = int(len(image_files) * SPLIT_RATIO)
    train_files = set(image_files[:split_idx])
    test_files = set(image_files[split_idx:])

    print(f"Splitting {len(image_files)} images into {len(train_files)} train and {len(test_files)} test")

    # Copy image files
    for fname in train_files:
        shutil.copy(os.path.join(image_dir, fname), os.path.join(TRAIN_DIR, fname))
    for fname in test_files:
        shutil.copy(os.path.join(image_dir, fname), os.path.join(TEST_DIR, fname))

    # Split captions.txt
    captions_src = os.path.join(RAW_DATA_DIR, "captions.txt")
    train_captions = []
    test_captions = []
    with open(captions_src, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            if '#0 ' in line:
                image, _ = line.split('#0 ', 1)
            elif ',' in line:
                image, _ = line.split(',', 1)
            else:
                continue
            if image in train_files:
                train_captions.append(line)
            elif image in test_files:
                test_captions.append(line)

    with open(os.path.join(TRAIN_DIR, "captions.txt"), "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(train_captions)

    with open(os.path.join(TEST_DIR, "captions.txt"), "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(test_captions)

    print(f"Dataset split completed. Preprocessed data is available at: {PREPROCESSED_DATA_DIR}")

def main():
    try:
        download_dataset()
    except Exception as e:
        print(f"Error during dataset preprocessing: {e}")
        exit(1)

if __name__ == "__main__":
    main()