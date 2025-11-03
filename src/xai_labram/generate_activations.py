import torch
import numpy as np
import json
from dotenv import load_dotenv
import os
import pickle
from tqdm import tqdm

# Import our validated ActivationExtractor
from src.xai_labram.activation_extractor import ActivationExtractor

# --- Configuration ---
CHECKPOINT_PATH = 'models/checkpoints/finetune_circling_v5/checkpoint-best.pth'
# Define the layers (bottlenecks) we want to test
TARGET_LAYERS = [0,1,2,3,4,5,6,7,8,9,10,11] 
# ---------------------

def load_eeg_from_file(filepath):
    """Helper function to load a single EEG sample from a .pkl file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        return torch.from_numpy(data['X'])

def process_and_save_activations(extractor, file_paths, dataset_name, output_dir):
    """
    Uses the extractor to get activations for all files in a list and 
    saves them to disk, organized by layer.
    """
    # This dictionary will store the activations, keyed by layer_id
    # e.g., { 3: [act1, act2, ...], 7: [act1, act2, ...], 11: [act1, act2, ...] }
    layer_activations = {layer: [] for layer in TARGET_LAYERS}

    print(f"Processing {dataset_name} dataset ({len(file_paths)} samples)...")
    for filepath in tqdm(file_paths):
        try:
            # 1. Load the raw EEG data
            eeg_tensor = load_eeg_from_file(filepath)
            
            # 2. Get activations from all target layers in one pass
            activations = extractor.get_activations(eeg_tensor, layer_ids=TARGET_LAYERS)
            
            # 3. Store the numpy arrays
            for layer_id, act_vector in activations.items():
                layer_activations[layer_id].append(act_vector)
                
        except Exception as e:
            print(f"Warning: Skipping file {filepath} due to error: {e}")

    # 4. Save the activations to disk
    # We save one file per layer for easy access
    for layer_id, acts_list in layer_activations.items():
        # Convert the list of 1D arrays into a 2D numpy array (samples, features)
        acts_array = np.array(acts_list)
        
        output_filename = f"{dataset_name}_layer_{layer_id}.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(acts_array, f)
        print(f"Saved {acts_array.shape} activations to {output_path}")

def main():
    load_dotenv()
    
    # --- Paths ---
    dataset_root = os.getenv("CIRCLING_DATASET_PATH")
    manifest_dir = os.path.join(dataset_root, "tcav_sanity_check_data_test_split")
    
    # This is where we will save the new activation files
    output_dir = os.path.join(dataset_root, "tcav_sanity_check_activations")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Manifests ---
    with open(os.path.join(manifest_dir, 'concept_open.json'), 'r') as f:
        concept_files = json.load(f)
    with open(os.path.join(manifest_dir, 'random_closed1.json'), 'r') as f:
        random_files = json.load(f)
    with open(os.path.join(manifest_dir, 'target_open.json'), 'r') as f:
        target_files = json.load(f)

    # --- Initialize Extractor ---
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ActivationExtractor(CHECKPOINT_PATH, device=device_str)

    # --- Process all three datasets ---
    process_and_save_activations(extractor, concept_files, "concept_open", output_dir)
    process_and_save_activations(extractor, random_files, "random_closed1", output_dir)
    process_and_save_activations(extractor, target_files, "target_open", output_dir)
    
    print("\nActivation generation complete.")
    print(f"All activation files saved in: {output_dir}")
    print("Ready for Part B: Training the CAVs.")

if __name__ == '__main__':
    main()

# python -m src.xai_labram.generate_activations