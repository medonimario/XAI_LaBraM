import mne
import os
import pickle
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# --- Configuration ---

# 1. Paths
INPUT_DIR = "/Volumes/T9/Circling_dataset/Epochs_labeled"
OUTPUT_DIR = "/Volumes/T9/Circling_dataset/tcav_data_pools"

# 2. Concept Parameters
N_EXAMPLES_PER_CONCEPT = 100  # Total samples for each concept pool
TARGET_SFREQ = 200.0

# 3. Concepts to build (band, channel, type)
# CONCEPTS_TO_DEFINE = [
#     ('alpha', 'C3', 'low'),
#     ('alpha', 'C3', 'high'),
#     ('alpha', 'C4', 'low'),
#     ('alpha', 'C4', 'high'),
#     ('beta', 'C3', 'low'),
#     ('beta', 'C3', 'high'),
# ]

# --- Model Channel Definition ---
# This list *must* match the 64-channel order expected by your model
# (Taken from your activation_extractor.py and make_MirrorGame_leadfollow.py)
STANDARD_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
]

CONCEPTS_TO_DEFINE = [(band, chan, type) for band in ['alpha', 'beta']
                      for chan in STANDARD_CHANNELS
                        for type in ['low', 'high']]

# --- Helper Function: Process and Save a Single Epoch ---
def process_and_save_epoch(epoch_data, subject_id, epoch_index, save_dir, label):
    """
    Takes a single MNE epoch object, processes it, and saves it as a .pkl file.
    """
    try:
        # 1. Resample to target frequency
        epoch_resampled = epoch_data.copy().resample(sfreq=TARGET_SFREQ)
        
        # 2. Drop the 'Status' channel
        epoch_resampled.drop_channels(['Status'])
        
        # 3. Re-order to match the model's expected 64 channels
        epoch_reordered = epoch_resampled.reorder_channels(STANDARD_CHANNELS)
        
        # 4. Get the data as a numpy array
        # .get_data() returns (n_epochs, n_chans, n_samples), so we squeeze it
        # We also convert from Volts (MNE default) to microvolts (like in your script)
        data = epoch_reordered.get_data(units='uV').squeeze()
        
        # 5. Verify shape (should be (64, 800) for 4s @ 200Hz)
        
        # --- FIX ---
        # Don't use epoch_data.tmax, which is the time of the last sample (e.g., 3.996s).
        # We must use the *true duration* of the original epoch, which is n_samples / sfreq.
        # This will be exactly 4.0s (e.g., 1024 samples / 256 Hz = 4.0s).
        
        original_duration_sec = 4.0  # Since all epochs are 4 seconds long
        expected_samples = int(original_duration_sec * TARGET_SFREQ)
        
        # Now, expected_samples will be int(4.0 * 200.0) = 800.
        # --- End Fix ---

        if data.shape != (len(STANDARD_CHANNELS), expected_samples):
            print(f"  [Warn] Subject {subject_id}, Epoch {epoch_index}: Skipped due to unexpected shape {data.shape}, should be {(len(STANDARD_CHANNELS), expected_samples)}.")
            return False

        # 6. Create save path and save .pkl file
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{subject_id}_epoch{epoch_index}.pkl"
        save_path = os.path.join(save_dir, filename)
        
        pkl_data = {"X": data, "y": label}
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_data, f)
            
        return True
            
    except Exception as e:
        print(f"  [Error] Subject {subject_id}, Epoch {epoch_index}: Failed processing. {e}")
        return False
    
# --- Main Function ---

def create_datasets():
    print("--- Starting Concept Dataset Creation ---")
    
    # Find all labeled .fif files
    all_fif_files = glob.glob(os.path.join(INPUT_DIR, "pair*epo.fif"))
    if not all_fif_files:
        print(f"Error: No .fif files found in {INPUT_DIR}. Stopping.")
        return

    # Get a list of all unique subjects (e.g., 'pair2010_a', 'pair2010_b')
    all_subjects = sorted([os.path.basename(f).split('-epo.fif')[0] for f in all_fif_files])
    n_total_subjects = len(all_subjects)
    print(f"Found {len(all_fif_files)} .fif files for {n_total_subjects} total subjects.")

    # --- 1. Create the 'random_pool' (All epochs, label 0) ---
    # print("\n--- Phase 1: Creating 'random_pool' (Label 0) ---")
    # random_pool_dir = os.path.join(OUTPUT_DIR, "random_pool")
    # random_epochs_saved = 0
    
    # for fif_file in tqdm(all_fif_files, desc="Processing files for random_pool"):
    #     subject_id = os.path.basename(fif_file).split('-epo.fif')[0]
    #     try:
    #         epochs = mne.read_epochs(fif_file, preload=True)
            
    #         for i in range(len(epochs)):
    #             # Get the single epoch object
    #             epoch_data = epochs[i]
                
    #             # Process and save this epoch
    #             success = process_and_save_epoch(
    #                 epoch_data,
    #                 subject_id=subject_id,
    #                 epoch_index=i,
    #                 save_dir=random_pool_dir,
    #                 label=0  # Label 0 for all epochs in the random pool
    #             )
    #             if success:
    #                 random_epochs_saved += 1
                    
    #     except Exception as e:
    #         print(f"Error loading file {fif_file}: {e}")

    # print(f"\nPhase 1 Complete: Saved {random_epochs_saved} epochs to {random_pool_dir}")

    # --- 2. Create Concept Pools (Systematic Sampling, Label 1) ---
    print("\n--- Phase 2: Creating Concept Pools (Label 1) ---")
    
    # Calculate how many epochs to get from each subject
    # We use max(1, ...) to ensure we get at least one from each subject
    n_epochs_per_subject = max(1, int(np.ceil(N_EXAMPLES_PER_CONCEPT / n_total_subjects)))
    print(f"Targeting {N_EXAMPLES_PER_CONCEPT} total examples per concept.")
    print(f"Will sample top {n_epochs_per_subject} epochs from each of {n_total_subjects} subjects.")

    all_subject_metadata = []
    print("Loading all metadata...")
    for fif_file in all_fif_files:
        try:
            epochs = mne.read_epochs(fif_file, preload=False)
            subject_id = os.path.basename(fif_file).split('-epo.fif')[0]
            
            # Store metadata with its source file and subject
            all_subject_metadata.append({
                "subject_id": subject_id,
                "fif_file": fif_file,
                "metadata": epochs.metadata
            })
        except Exception as e:
            print(f"Error reading metadata from {fif_file}: {e}")

    # Now, loop through the concepts we want to define
    for (band, chan, type) in CONCEPTS_TO_DEFINE:
        concept_name = f"concept_{band}_{chan}_{type}"
        concept_col = f"z_{band}_{chan}"
        concept_dir = os.path.join(OUTPUT_DIR, concept_name)
        
        print(f"\nCreating concept: {concept_name}")
        
        concept_epochs_saved = 0
        
        # Iterate through each subject's metadata
        for subject_info in tqdm(all_subject_metadata, desc=f"Building {concept_name}"):
            
            # --- THIS IS THE FIX ---
            # Reset the index so it goes from 0 to N-1, matching the
            # MNE.Epochs object we are about to load.
            metadata = subject_info['metadata'].copy().reset_index(drop=True)
            # --- END FIX ---
            
            subject_id = subject_info['subject_id']
            
            if concept_col not in metadata.columns:
                print(f"  [Warn] {subject_id}: Column {concept_col} not found. Skipping.")
                continue
                
            # Sort metadata to find the epochs we want
            if type == 'low':
                # Sort ascending, take the top 'n_epochs_per_subject'
                sorted_meta = metadata.nsmallest(n_epochs_per_subject, concept_col)
            elif type == 'high':
                # Sort descending, take the top 'n_epochs_per_subject'
                sorted_meta = metadata.nlargest(n_epochs_per_subject, concept_col)
            else:
                continue
            
            # Get the indices of these epochs
            # (These will now be positional indices, e.g., 0, 5, 12... 55)
            epoch_indices_to_load = sorted_meta.index.tolist()
            
            if not epoch_indices_to_load:
                print(f"  [Info] {subject_id}: No epochs found for this concept.")
                continue

            # Load the corresponding .fif file *once*
            try:
                epochs = mne.read_epochs(subject_info['fif_file'], preload=True)
            except Exception as e:
                print(f"  [Error] {subject_id}: Could not load {subject_info['fif_file']}. {e}")
                continue

            # Process and save only the selected epochs
            for epoch_idx in epoch_indices_to_load:
                # This line will now work, as epoch_idx will be e.g. 5,
                # not 57.
                epoch_data = epochs[epoch_idx] 
                
                success = process_and_save_epoch(
                    epoch_data,
                    subject_id=subject_id,
                    epoch_index=epoch_idx, # Using the positional index
                    save_dir=concept_dir,
                    label=1  # Label 1 for all concept epochs
                )
                if success:
                    concept_epochs_saved += 1

        print(f"Concept '{concept_name}' complete: Saved {concept_epochs_saved} epochs to {concept_dir}")

    print("\n--- All processing complete. ---")

if __name__ == "__main__":
    create_datasets()