import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

# Standard channel order for model compatibility.
# Verify that these channels exist in your preprocessed data.
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
    Processes a single EEGLAB epoch file (.set).
    Since each file corresponds to one condition, the label is passed in.
    It saves each epoch individually as a pickle file.
    """
    filepath, dump_folder, chOrder_standard, label = params
    
    print(f"Processing {os.path.basename(filepath)} with label {label}")
    try:
        # Load the pre-epoched EEGLAB data file
        epochs = mne.io.read_epochs_eeglab(filepath, verbose=False)
        epochs.set_eeg_reference('average', projection=True)
        epochs.crop(tmin=0.0)

        # Resample to 200 Hz if not already
        if epochs.info['sfreq'] != 200:
            epochs.resample(200, n_jobs=1, verbose=False)

        # Optional but recommended: reorder channels for consistency.
        if chOrder_standard and all(ch in epochs.ch_names for ch in chOrder_standard):
            epochs.reorder_channels(chOrder_standard)
        else:
            print(f"Warning: Not all standard channels found in {filepath}. Skipping reorder.")
        
        # Get all epoch data at once for efficiency
        data = epochs.get_data(units='uV')
        print(f"Data shape for {os.path.basename(filepath)}: {data.shape}")
        
        base_filename = os.path.basename(filepath).split('.set')[0]

        # Loop through each individual epoch contained in the file
        for i in range(len(epochs)):
            # The label is the same for all epochs in this file
            dump_path = os.path.join(
                dump_folder, f"{base_filename}_epoch{i}.pkl"
            )
            
            # The data for this single epoch is at index i
            epoch_data = data[i, :, :]
            
            pickle.dump(
                {"X": epoch_data, "y": label},
                open(dump_path, "wb"),
            )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(f"{filepath} - {e}\n")

def get_subject_from_filename(filename):
    """
    Extracts the subject ID from a filename like 's_101_Coordination.set'.
    It captures the number following 's_'.
    """
    match = re.search(r's_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    # Set path to your dataset folder (containing the .set files)
    raw_dataset_path = os.getenv("MG_DATASET_PATH") + "preprocessed/"
    
    # Filter for .set files and only include 'Solo' or 'Coordination' conditions
    all_files = [
        f for f in os.listdir(raw_dataset_path) 
        if f.endswith(".set") and ('Solo' in f or 'Spontaneous' in f)
    ]
    
    # Group all relevant files by their subject ID
    subject_files = defaultdict(list)
    for f in all_files:
        subject_id = get_subject_from_filename(f)
        if subject_id:
            subject_files[subject_id].append(f)
        else:
            print(f"Warning: Could not find subject ID for file {f}")

    # Split by subject ID for train, val, test
    all_subject_ids = sorted(subject_files.keys())
    np.random.seed(42) # for reproducibility
    np.random.shuffle(all_subject_ids)

    # Splitting subjects: ~80% train, ~10% validation, ~10% test
    train_split = int(len(all_subject_ids) * 0.8)
    val_split = int(len(all_subject_ids) * 0.9)

    train_subject_ids = all_subject_ids[:train_split]
    val_subject_ids = all_subject_ids[train_split:val_split]
    test_subject_ids = all_subject_ids[val_split:]

    print(f"Total subjects: {len(all_subject_ids)}")
    print(f"Train subjects ({len(train_subject_ids)}): {train_subject_ids}")
    print(f"Validation subjects ({len(val_subject_ids)}): {val_subject_ids}")
    print(f"Test subjects ({len(test_subject_ids)}): {test_subject_ids}")

    # Create file lists based on subject splits
    train_files = [f for sid in train_subject_ids for f in subject_files[sid]]
    val_files = [f for sid in val_subject_ids for f in subject_files[sid]]
    test_files = [f for sid in test_subject_ids for f in subject_files[sid]]

    # Set your processed data path
    final_root = os.getenv("MG_DATASET_PATH") # Make sure to set this in your .env file
    
    # Create the train, val, test sample folders
    train_dump_folder = os.path.join(final_root, "processed_solo-spont", "train")
    val_dump_folder = os.path.join(final_root, "processed_solo-spont", "val")
    test_dump_folder = os.path.join(final_root, "processed_solo-spont", "test")

    os.makedirs(train_dump_folder, exist_ok=True)
    os.makedirs(val_dump_folder, exist_ok=True)
    os.makedirs(test_dump_folder, exist_ok=True)

    # Prepare parameters for the processing function
    parameters = []
    
    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            filepath = os.path.join(raw_dataset_path, f)
            # Determine label from filename: 0 for Solo, 1 for Spontaneous
            label = 1 if 'Spontaneous' in f else 0
            params.append([filepath, dump_folder, standard_channels, label])
        return params

    parameters.extend(prepare_params(train_files, train_dump_folder))
    parameters.extend(prepare_params(val_files, val_dump_folder))
    parameters.extend(prepare_params(test_files, test_dump_folder))

    # Use a number of processes suitable for your machine
    # Set to 1 if you encounter issues, to debug in serial mode.
    num_processes = 8
    with Pool(processes=num_processes) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")