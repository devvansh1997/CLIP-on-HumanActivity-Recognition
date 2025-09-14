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
from src.model import create_model
from transformers import AutoProcessor # Import AutoProcessor

def evaluate(config_path: str, checkpoint_path: str):
    """
    Evaluates a trained model checkpoint on the test dataset.
    """
    print(f"🚀 Starting evaluation for checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)

    # Determine class names from the test directory structure
    test_path = os.path.join(config['data']['path'], 'test')
    class_names = sorted([name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))])
    
    # Create the test dataloader
    print("Creating test dataloader...")
    test_loader = create_dataloader(config, split='test')

    # Load the model and the fine-tuned weights
    print("Loading model from checkpoint...")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truth = []

    # Pre-compute text embeddings for all class names for efficiency
    with torch.no_grad():
        # --- FIX: Create the processor directly ---
        processor = AutoProcessor.from_pretrained(config['model']['name'])
        
        text_inputs = processor(text=class_names, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeds = model.clip.get_text_features(**text_inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    print("Running inference on the test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["pixel_values"].to(device)
            ground_truth_labels = batch["label_idx"]
            all_ground_truth.extend(ground_truth_labels.cpu().numpy())
            
            # Get image embeddings
            image_embeds = model.clip.get_image_features(pixel_values=images)
            image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

            # Calculate similarity against all class text embeddings
            similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
            
            # Get the top prediction
            predictions = similarity.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate and Print Metrics
    print("\n--- Evaluation Results ---")
    report = classification_report(
        all_ground_truth,
        all_predictions,
        target_names=class_names,
        labels=range(len(class_names)) # Explicitly provide all possible labels
    )
    print(report)

    # Generate and Display Confusion Matrix
    print("Generating confusion matrix...")
    labels_to_display = range(len(class_names))
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