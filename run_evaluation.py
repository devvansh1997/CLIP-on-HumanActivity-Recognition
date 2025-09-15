import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

# Import the necessary components
from src.utils import load_config
from src.dataset import create_dataloader
from src.model import create_model, CustomCLIP, SiglipForFineTuning
from transformers import AutoProcessor

def evaluate(config_path: str, checkpoint_path: str):
    """
    Evaluates a trained model checkpoint on the test dataset.
    This script is flexible and handles both CLIP and SigLIP models correctly.
    """
    print(f"🚀 Starting evaluation for checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)
    model_type = config['model'].get('type', 'clip')

    test_path = os.path.join(config['data']['path'], 'test')
    class_names = sorted([name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))])
    
    print("Creating test dataloader...")
    test_loader = create_dataloader(config, split='test')

    print("Loading model from checkpoint...")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        processor = AutoProcessor.from_pretrained(config['model']['name'])
        text_inputs = processor(text=class_names, return_tensors="pt", padding=True, truncation=True).to(device)
        
        if model_type == 'siglip':
            assert isinstance(model, SiglipForFineTuning)
            text_embeds = model.model.get_text_features(**text_inputs)
        else:
            assert isinstance(model, CustomCLIP)
            text_embeds = model.clip.get_text_features(**text_inputs)
        
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    print("Running inference on the test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["pixel_values"].to(device)
            ground_truth_labels = batch["label_idx"]
            all_ground_truth.extend(ground_truth_labels.cpu().numpy())
            
            if model_type == 'siglip':
                assert isinstance(model, SiglipForFineTuning)
                image_embeds = model.model.get_image_features(pixel_values=images)
            else:
                assert isinstance(model, CustomCLIP)
                image_embeds = model.clip.get_image_features(pixel_values=images)
            
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

            # --- CORRECTED LOGIC FOR PREDICTION ---
            if model_type == 'siglip':
                # For SigLIP, we use the raw similarity scores (logits) directly
                similarity = image_embeds @ text_embeds.T
            else:
                # For CLIP, we scale by the learnable temperature and use softmax
                similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)

            predictions = similarity.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # ... (The rest of the script for metrics and plotting is the same) ...
    print("\n--- Evaluation Results ---")
    labels_to_display = range(len(class_names))
    report = classification_report(all_ground_truth, all_predictions, target_names=class_names, labels=labels_to_display, zero_division=0)
    print(report)

    print("Generating confusion matrix...")
    cm = confusion_matrix(all_ground_truth, all_predictions, labels=labels_to_display)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the saved model checkpoint (.pt).")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)