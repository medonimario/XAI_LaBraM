import os
import pickle
import re # <-- MODIFICATION: To extract triad IDs from filenames
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
load_dotenv()
import shutil
from collections import defaultdict # <-- MODIFICATION

# It's good practice to keep a standard channel order for model compatibility.
# You should verify that these channels exist in your preprocessed data.
standard_channels = ['Fp1','Fpz','Fp2',
                     'AF7','AF3','AFz','AF4','AF8',
                     'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
                     'FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8',
                     'T7','C5','C3','C1','Cz','C2','C4','C6','T8',
                     'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8',
                     'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10',
                     'PO7','PO3','POz','PO4','PO8',
                     'O1','Oz','O2',
                     'Iz'
                    ]

def process_epoch_file(params):
    """
    Processes a single MNE epoch file. It filters for solo (T1P*) and
    trio (T3P*) conditions, assigns a class label based on the condition
    (0 for solo, 1 for trio), and saves each epoch individually.
    """
    filepath, dump_folder, chOrder_standard = params
    
    print("Processing", os.path.basename(filepath))
    try:
        # Load the pre-epoched data file
        epochs = mne.read_epochs(filepath, preload=True, verbose=False)

        # <-- MODIFICATION 1: Filter to keep only solo and trio conditions -->
        conditions_to_keep = [
            cond for cond in epochs.event_id.keys()
            if cond.startswith('T1P') or cond.startswith('T3P')
        ]
        
        # If there are relevant conditions in this file, select them
        if conditions_to_keep:
            epochs = epochs[conditions_to_keep]
        else:
            # If no solo or trio conditions are found, skip this file
            print(f"Info: No solo or trio conditions in {os.path.basename(filepath)}. Skipping.")
            return # Exit the function for this file

        epochs.crop(tmax=3.5)

        if epochs.info['sfreq'] != 200:
            epochs.resample(200, n_jobs=1)

        if chOrder_standard is not None and len(chOrder_standard) == len(epochs.ch_names):
            if all(ch in epochs.ch_names for ch in chOrder_standard):
                 epochs.reorder_channels(chOrder_standard)
            else:
                print(f"Warning: Not all standard channels found in {filepath}. Skipping reorder.")

        rev_event_id = {v: k for k, v in epochs.event_id.items()}
        data = epochs.get_data(units='uV')
        base_filename = os.path.basename(filepath).split('-epo.fif')[0]

        for i in range(len(epochs)):
            event_code = epochs.events[i, 2]
            condition_name = rev_event_id[event_code]
            
            # <-- MODIFICATION 2: Change labeling to solo vs. trio -->
            # Class 0 for solo (starts with 'T1P')
            # Class 1 for trio (starts with 'T3P')
            if condition_name.startswith('T1P'):
                label = 0
            elif condition_name.startswith('T3P'):
                label = 1
            else:
                # This should not be reached due to the filtering above,
                # but it's a safeguard.
                continue

            dump_path = os.path.join(
                dump_folder, f"{base_filename}_condition_{condition_name}_epoch{i}.pkl"
            )
            
            epoch_data = data[i, :, :]
            
            pickle.dump(
                {"X": epoch_data, "y": label},
                open(dump_path, "wb"),
            )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(filepath + "\n")

def get_triad_from_filename(filename):
    """
    Extracts the triad ID from a filename like '301A_FG_...-epo.fif'.
    It captures the number at the start of the string.
    """
    # <-- MODIFICATION: Updated regex to match your filenames
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    # Set path to your dataset folder (containing the 92 .fif files)
    raw_dataset_path = os.getenv("FG_DATASET_PATH") + "preprocessed/"
    
    # Group all epoch files by their triad ID
    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("-epo.fif")]
    
    triad_files = defaultdict(list)
    for f in all_files:
        triad_id = get_triad_from_filename(f)
        if triad_id:
            triad_files[triad_id].append(f)
        else:
            print(f"Warning: Could not find triad ID for file {f}")

    # Split by triad ID for train, val, test
    all_triad_ids = sorted(triad_files.keys())
    np.random.seed(42) # for reproducibility
    np.random.shuffle(all_triad_ids)

    # Splitting triads: ~80% train, ~10% validation, ~10% test
    train_split = int(len(all_triad_ids) * 0.8)
    val_split = int(len(all_triad_ids) * 0.9)

    train_triad_ids = all_triad_ids[:train_split]
    val_triad_ids = all_triad_ids[train_split:val_split]
    test_triad_ids = all_triad_ids[val_split:]

    print(f"Total triads: {len(all_triad_ids)}")
    print(f"Train triads ({len(train_triad_ids)}): {train_triad_ids}")
    print(f"Validation triads ({len(val_triad_ids)}): {val_triad_ids}")
    print(f"Test triads ({len(test_triad_ids)}): {test_triad_ids}")

    # Create file lists based on triad splits
    train_files = [f for tid in train_triad_ids for f in triad_files[tid]]
    val_files = [f for tid in val_triad_ids for f in triad_files[tid]]
    test_files = [f for tid in test_triad_ids for f in triad_files[tid]]

    # Set your processed data path
    final_root = os.getenv("FG_DATASET_PATH")
    
    # Create the train, val, test sample folders
    train_dump_folder = os.path.join(final_root, "processed_group", "train")
    val_dump_folder = os.path.join(final_root, "processed_group", "val")
    test_dump_folder = os.path.join(final_root, "processed_group", "test")

    os.makedirs(train_dump_folder, exist_ok=True)
    os.makedirs(val_dump_folder, exist_ok=True)
    os.makedirs(test_dump_folder, exist_ok=True)

    # Prepare parameters for the processing function
    parameters = []
    
    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            filepath = os.path.join(raw_dataset_path, f)
            # The label is now determined inside the processing function
            params.append([filepath, dump_folder, standard_channels])
        return params

    parameters.extend(prepare_params(train_files, train_dump_folder))
    parameters.extend(prepare_params(val_files, val_dump_folder))
    parameters.extend(prepare_params(test_files, test_dump_folder))

    # Use a number of processes suitable for your machine
    with Pool(processes=8) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")