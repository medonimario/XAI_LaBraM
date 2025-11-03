import torch
import numpy as np
import json
from dotenv import load_dotenv
import os
import pickle
from tqdm import tqdm
import argparse

# Import our validated ActivationExtractor
from src.xai_labram.activation_extractor import ActivationExtractor

# Define the layers (bottlenecks) we want to test
# We can make this a script argument for more flexibility
TARGET_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def load_eeg_from_file(filepath):
    """Helper function to load a single EEG sample from a .pkl file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return torch.from_numpy(data['X'])
    except FileNotFoundError:
        print(f"Error: File not found {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_and_save_activations(extractor, file_paths, dataset_name, output_dir):
    """
    Uses the extractor to get activations for all files in a list and
    saves them to disk, organized by layer.
    """
    # This dictionary will store the activations, keyed by layer_id
    # e.g., { 3: [act1, act2, ...], 7: [act1, act2, ...], 11: [act1, act2, ...] }
    layer_activations = {layer: [] for layer in TARGET_LAYERS}

    print(f"\nProcessing '{dataset_name}' dataset ({len(file_paths)} samples)...")
    for filepath in tqdm(file_paths, desc=f"Extracting {dataset_name}"):
        # 1. Load the raw EEG data
        eeg_tensor = load_eeg_from_file(filepath)
        if eeg_tensor is None:
            continue
            
        try:
            # 2. Get activations from all target layers in one pass
            activations = extractor.get_activations(eeg_tensor, layer_ids=TARGET_LAYERS)
            
            # 3. Store the numpy arrays
            for layer_id, act_vector in activations.items():
                if act_vector is not None and act_vector.size > 0:
                    layer_activations[layer_id].append(act_vector)
                else:
                    print(f"Warning: Got empty activation for layer {layer_id} in file {filepath}")
                
        except Exception as e:
            print(f"Warning: Skipping file {filepath} due to error: {e}")

    # 4. Save the activations to disk
    # We save one file per layer for easy access
    print(f"Saving activations for '{dataset_name}'...")
    for layer_id, acts_list in layer_activations.items():
        if not acts_list:
            print(f"Warning: No valid activations processed for {dataset_name}, layer {layer_id}. Skipping save.")
            continue
            
        # Convert the list of 1D arrays into a 2D numpy array (samples, features)
        acts_array = np.array(acts_list)
        
        output_filename = f"{dataset_name}_layer_{layer_id}.pkl"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(acts_array, f)
            print(f"  Saved {acts_array.shape} activations to {output_path}")
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

def main(args):
    load_dotenv()
    
    # --- Paths ---
    manifest_dir = args.manifest_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load Manifests ---
    print(f"Loading manifests from: {manifest_dir}")
    try:
        with open(os.path.join(manifest_dir, 'concept_set.json'), 'r') as f:
            concept_files = json.load(f)
        with open(os.path.join(manifest_dir, 'target_class_set.json'), 'r') as f:
            target_files = json.load(f)
        with open(os.path.join(manifest_dir, 'random_sets.json'), 'r') as f:
            random_sets_all_runs = json.load(f) # This is a list of lists
    except FileNotFoundError as e:
        print(f"Error: Manifest file not found. Did you run the preparation script?")
        print(f"{e}")
        return
        
    print(f"Found {len(concept_files)} concept files.")
    print(f"Found {len(target_files)} target files.")
    print(f"Found {len(random_sets_all_runs)} runs of random sets.")

    # --- Initialize Extractor ---
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing ActivationExtractor on {device_str}...")
    try:
        extractor = ActivationExtractor(args.checkpoint_path, device=device_str)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Process all datasets ---
    
    # 1. Process the (single) Concept Set
    process_and_save_activations(extractor, concept_files, "concept_set", output_dir)
    
    # 2. Process the (single) Target Class Set
    process_and_save_activations(extractor, target_files, "target_class_set", output_dir)

    # 3. Process the (multiple) Random Sets
    # This is the new, crucial part
    print(f"\n--- Processing {len(random_sets_all_runs)} Random Runs ---")
    for i, random_file_list in enumerate(random_sets_all_runs):
        dataset_name = f"random_run_{i}"
        process_and_save_activations(extractor, random_file_list, dataset_name, output_dir)
    
    print("\nActivation generation complete.")
    print(f"All activation files saved in: {output_dir}")
    print("Ready for Part C: Training the CAVs and running TCAV.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save model activations for TCAV.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pth finetuned model checkpoint.")
    parser.add_argument("--manifest_dir", type=str, required=True,
                        help="Directory containing the JSON manifest files (concept_set.json, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output .pkl activation files.")
    # You could add --target_layers here if you want to make it configurable
    
    args = parser.parse_args()
    
    # Update TARGET_LAYERS from args if you implement it
    # global TARGET_LAYERS
    # TARGET_LAYERS = [int(l) for l in args.target_layers.split(',')]
    
    main(args)

# Example command to run this script:
# python -m src.xai_labram.02_generate_activations \
#     --checkpoint_path models/checkpoints/finetune_circling_v5/checkpoint-best.pth \
#     --manifest_dir /work3/s204684/Thesis/circling_eeg/tcav/sanity_check_open \
#     --output_dir /work3/s204684/Thesis/circling_eeg/tcav/sanity_check_open/activations