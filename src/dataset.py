import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from PIL import Image
import os
import random
from typing import List, Dict, Any

# --- Perfect Prompt Dictionary ---
# A dictionary mapping class names to a list of descriptive, varied prompts.
PROMPT_TEMPLATES = {
    "calling": [
        "a photo of a person making a phone call",
        "an image of someone talking on their phone",
        "a person holding a phone to their ear"
    ],
    "clapping": [
        "a photo of a person clapping their hands",
        "an image of someone applauding",
        "a person in the act of clapping"
    ],
    "cycling": [
        "a photo of a person riding a bicycle",
        "an image of someone cycling",
        "a person on a bike"
    ],
    "dancing": [
        "a photo of a person dancing",
        "an image of someone moving to music",
        "a person showing dance moves"
    ],
    "drinking": [
        "a photo of a person drinking from a cup or bottle",
        "an image of someone taking a sip of a beverage",
        "a person drinking something"
    ],
    "eating": [
        "a photo of a person eating food",
        "an image of someone having a meal",
        "a person consuming food"
    ],
    "fighting": [
        "a photo of people fighting or arguing",
        "an image of a physical confrontation",
        "individuals engaged in a fight"
    ],
    "hugging": [
        "a photo of people hugging each other",
        "an image of an embrace between two individuals",
        "two people hugging"
    ],
    "laughing": [
        "a photo of a person laughing heartily",
        "an image of someone showing joy and amusement",
        "a person captured mid-laugh"
    ],
    "listening_to_music": [
        "a photo of a person listening to music with headphones",
        "an image of someone enjoying music",
        "a person with earbuds or headphones on"
    ],
    "running": [
        "a photo of a person running or jogging",
        "an image of someone in the act of running",
        "a person running outdoors"
    ],
    "sitting": [
        "a photo of a person sitting down",
        "an image of someone seated on a chair or surface",
        "a person in a sitting position"
    ],
    "sleeping": [
        "a photo of a person sleeping",
        "an image of someone resting or asleep",
        "a person lying down and sleeping"
    ],
    "texting": [
        "a photo of a person texting on their phone",
        "an image of someone typing on a smartphone",
        "a person using their phone to send a message"
    ],
    "using_laptop": [
        "a photo of a person using a laptop computer",
        "an image of someone working on a laptop",
        "a person typing on their laptop"
    ]
}


class HARDataset(Dataset):
    """
    PyTorch Dataset for Human Activity Recognition.
    """
    def __init__(self,
                 image_paths: List[str],
                 text_prompts: List[str],
                 labels: List[int],
                 processor):
        
        self.image_paths = image_paths
        self.text_prompts = text_prompts
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves and processes a single image-text pair and its label.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.text_prompts[idx]
        label = self.labels[idx]

        inputs = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_idx": torch.tensor(label, dtype=torch.long)
        }


def create_dataloader(config: Dict[str, Any], split: str) -> DataLoader:
    """
    Factory function to create the HAR dataloader for a specific split.
    """
    if split not in ['train', 'test']:
        raise ValueError(f"Split must be 'train' or 'test', but got {split}")

    processor = AutoProcessor.from_pretrained(config['model']['name'])
    data_split_path = os.path.join(config['data']['path'], split)

    all_image_paths = []
    all_text_prompts = []
    all_labels = [] # NEW: List to store integer labels

    # NEW: Create a mapping from class name to an integer index
    class_names = sorted([name for name in os.listdir(data_split_path) if os.path.isdir(os.path.join(data_split_path, name))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print(f"Loading data from: {data_split_path}")
    
    for class_name, label_idx in class_to_idx.items():
        if class_name not in PROMPT_TEMPLATES:
            print(f"Warning: Skipping folder '{class_name}' as it's not in PROMPT_TEMPLATES.")
            continue

        class_dir = os.path.join(data_split_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(class_dir, img_name))
                prompt = random.choice(PROMPT_TEMPLATES[class_name])
                all_text_prompts.append(prompt)
                all_labels.append(label_idx) # NEW: Add the integer label

    if not all_image_paths:
        raise FileNotFoundError(f"No images found in the directory: {data_split_path}.")
    
    if config['training'].get('dev_run', False):
        print(f"🚨 Running in DEV MODE on a subset of {config['training']['dev_run_size']} samples.")
        subset_size = min(len(all_image_paths), config['training']['dev_run_size'])
        all_image_paths = all_image_paths[:subset_size]
        all_text_prompts = all_text_prompts[:subset_size]
        all_labels = all_labels[:subset_size]
    
    print(f"Found {len(all_image_paths)} images and prompts for the '{split}' split across {len(class_to_idx)} classes.")
    
    dataset = HARDataset(
        image_paths=all_image_paths,
        text_prompts=all_text_prompts,
        labels=all_labels, # NEW: Pass labels to the dataset
        processor=processor
    )
    
    return DataLoader(
        dataset, 
        batch_size=config['training']['mini_batch_size'], 
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )