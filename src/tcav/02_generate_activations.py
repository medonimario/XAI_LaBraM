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
    
    # --- MODIFIED: Initialize all file lists as empty ---
    concept_files = []
    target_files = []
    random_sets_all_runs = []

    try:
        # 1. Concept (Always required)
        with open(os.path.join(manifest_dir, 'concept_set.json'), 'r') as f:
            concept_files = json.load(f)
        print(f"Found {len(concept_files)} concept files.")
    except FileNotFoundError as e:
        print(f"Error: Required manifest 'concept_set.json' not found.")
        print(f"{e}")
        return

    # --- MODIFIED: Conditionally load target ---
    if not args.skip_target_set:
        try:
            with open(os.path.join(manifest_dir, 'target_class_set.json'), 'r') as f:
                target_files = json.load(f)
            print(f"Found {len(target_files)} target files.")
        except FileNotFoundError as e:
            print(f"Error: 'target_class_set.json' not found. (Use --skip_target_set to ignore).")
            print(f"{e}")
            return
    else:
        print("Skipping target_class_set.json loading.")

    # --- MODIFIED: Conditionally load random ---
    if not args.skip_random_sets:
        try:
            with open(os.path.join(manifest_dir, 'random_sets.json'), 'r') as f:
                random_sets_all_runs = json.load(f) # This is a list of lists
            print(f"Found {len(random_sets_all_runs)} runs of random sets.")
        except FileNotFoundError as e:
            print(f"Error: 'random_sets.json' not found. (Use --skip_random_sets to ignore).")
            print(f"{e}")
            return
    else:
        print("Skipping random_sets.json loading.")

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
    # (Always runs)
    if not concept_files:
        print("\nWarning: Concept file list is empty. Nothing to process for 'concept_set'.")
    else:
        process_and_save_activations(extractor, concept_files, "concept_set", output_dir)
    
    # 2. Process the (single) Target Class Set
    # --- MODIFIED ---: Conditional processing
    if not args.skip_target_set:
        if not target_files:
            print("\nWarning: Target file list is empty. Nothing to process for 'target_class_set'.")
        else:
            process_and_save_activations(extractor, target_files, "target_class_set", output_dir)
    else:
        print("\nSkipping activation generation for Target Class set.")

    # 3. Process the (multiple) Random Sets
    # --- MODIFIED ---: Conditional processing
    if not args.skip_random_sets:
        if not random_sets_all_runs:
                print("\nWarning: Random sets list is empty. Nothing to process for 'random_sets'.")
        else:
            print(f"\n--- Processing {len(random_sets_all_runs)} Random Runs ---")
            for i, random_file_list in enumerate(random_sets_all_runs):
                dataset_name = f"random_run_{i}"
                process_and_save_activations(extractor, random_file_list, dataset_name, output_dir)
    else:
        print("\nSkipping activation generation for Random sets.")
    
    print("\nActivation generation complete.")
    print(f"All generated activation files saved in: {output_dir}")
    print("Ready for Part C: Training the CAVs and running TCAV.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save model activations for TCAV.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pth finetuned model checkpoint.")
    parser.add_argument("--manifest_dir", type=str, required=True,
                        help="Directory containing the JSON manifest files (concept_set.json, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output .pkl activation files.")
    
    # --- NEW ---: Optional skip flags
    parser.add_argument("--skip_target_set", action='store_true',
                        help="Do not load or process activations for the target_class_set.json.")
    parser.add_argument("--skip_random_sets", action='store_true',
                        help="Do not load or process activations for the random_sets.json.")
    
    args = parser.parse_args()
    
    main(args)