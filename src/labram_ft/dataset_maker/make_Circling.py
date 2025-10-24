import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
import shutil
from collections import defaultdict
import math # <-- MODIFICATION: Added for the segmentation function

load_dotenv()

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

# <-- MODIFICATION: The user-provided segmentation function is included here.
def segment_epochs(full_epochs, condition, segm_length_sec):
    """
    Segments longer epochs into multiple shorter, consecutive epochs.
    """
    # Calculate how many segments can be created from the duration of the epoch
    # The original sampling frequency is used for this calculation before resampling.
    duration = len(full_epochs[condition].get_data()[0][0]) / full_epochs.info['sfreq']
    n_segm = math.floor(duration / segm_length_sec)
    
    epochs_list = []
    x = 0
    k = 0  # A time shifter to make sure all shorter epochs have the same times
    
    for i in range(n_segm):
        # Create a fresh copy for each crop operation
        cond_epochs = mne.EpochsArray(data=full_epochs[condition].get_data(), info=full_epochs.info, verbose=False)
        # Crop to the current segment
        shorter_epochs = cond_epochs.crop(tmin=x, tmax=x + segm_length_sec, include_tmax=False, verbose=False)
        print(f"Segment {i}: {shorter_epochs.get_data().shape}")
        # Shift the time axis to start at 0
        shorter_epochs.shift_time(k, relative=True)
        epochs_list.append(shorter_epochs)
        
        k = -x - segm_length_sec
        x = x + segm_length_sec
        
    # Concatenate all the shorter epochs into one object
    conc_object = mne.concatenate_epochs(epochs_list, add_offset=True, verbose=False)
    # Reset indices by creating a final EpochsArray
    final_object = mne.EpochsArray(data=conc_object.get_data(), info=conc_object.info, verbose=False)
    return final_object

# <-- MODIFICATION: The main processing function has been heavily modified to include the new steps.
def process_epoch_file(params):
    """
    Processes a single MNE epoch file for the 'pairs' dataset.
    It loads, crops, segments, cleans, resamples, and saves each resulting 
    6-second epoch as a pickle file.
    """
    filepath, dump_folder, chOrder_standard, bad_epochs_dict = params
    
    base_filename = os.path.basename(filepath).split('_ave_epo.fif')[0]
    print("Processing", base_filename)

    try:
        # --- 1. Load Data ---
        epochs = mne.read_epochs(filepath, preload=True, verbose=False)
        
        # --- 2. Remove Practice Trials ---
        pair_n_str = re.search(r'pair(\d+)', base_filename).group(1)
        practice_trial_cut = 12 if pair_n_str == '2030' else 6
        
        # --- 3. Initial Preprocessing ---
        # Crop first 2 seconds and then remove practice trials from the start
        epochs = epochs.crop(tmin=2, verbose=False)[practice_trial_cut:]

        # Conditions to process
        conditions_to_process = {'open': 1, 'closed1': 0} # condition -> label

        for condition, label in conditions_to_process.items():
            if condition not in epochs.event_id:
                continue # Skip if this file doesn't have the condition

            # --- 4. Segment into 6-second chunks ---
            segmented_epochs = segment_epochs(full_epochs=epochs, condition=condition, segm_length_sec=6)
            
            # --- 5. Remove Bad Epochs ---
            # Construct the key for the bad epochs dictionary (e.g., '2010_a_open')
            bad_epoch_key = f"{pair_n_str}_{base_filename[-1]}_{condition}" # TODO: Verify if base_filename[-1] is correct
            segments_to_remove = bad_epochs_dict.get(bad_epoch_key, [])
            
            # Get data as a numpy array and remove bad segments
            segmented_data = segmented_epochs.get_data(units='uV')
            if segments_to_remove:
                cleaned_data = np.delete(segmented_data, obj=segments_to_remove, axis=0)
            else:
                cleaned_data = segmented_data

            # --- 6. Process and Save Each Cleaned Segment ---
            for i in range(cleaned_data.shape[0]):
                # Create a temporary MNE object for the single segment to use MNE's tools
                single_epoch_data = cleaned_data[i:i+1, :, :]
                temp_epoch_obj = mne.EpochsArray(single_epoch_data, epochs.info, verbose=False)

                # Resample to 200 Hz if needed
                if temp_epoch_obj.info['sfreq'] != 200:
                    temp_epoch_obj.resample(200, n_jobs=1, verbose=False)

                # Reorder channels for consistency
                if chOrder_standard is not None and len(chOrder_standard) == len(temp_epoch_obj.ch_names):
                    if all(ch in temp_epoch_obj.ch_names for ch in chOrder_standard):
                         temp_epoch_obj.reorder_channels(chOrder_standard)
                    else:
                        print(f"Warning: Not all standard channels found in {filepath}. Skipping reorder.")
                
                # Get the final processed data for this segment
                final_epoch_data = temp_epoch_obj.get_data()[0]

                # Define a descriptive output filename
                dump_path = os.path.join(
                    dump_folder, f"{base_filename}_condition_{condition}_segment{i}.pkl"
                )
                
                # Save the processed data and label to a pickle file
                pickle.dump(
                    {"X": final_epoch_data, "y": label},
                    open(dump_path, "wb"),
                )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(filepath + "\n")

# <-- MODIFICATION: Renamed and updated regex to extract pair ID.
def get_pair_from_filename(filename):
    """
    Extracts the pair ID from a filename like 'pair2010_a_ave_epo.fif'.
    """
    match = re.search(r'pair(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    # <-- MODIFICATION: Set paths for the new dataset and bad epochs dictionary.
    base_path = os.getenv("CIRCLING_DATASET_PATH")
    raw_dataset_path = os.path.join(base_path, "preprocessed/")
    bad_epochs_pickle_path = os.path.join(base_path, "dict_with_short_bad_epochs.pickle")

    # <-- MODIFICATION: Load the bad epochs dictionary once.
    print(f"Loading bad epochs dictionary from: {bad_epochs_pickle_path}")
    with open(bad_epochs_pickle_path, "rb") as f:
        bad_short_epochs = pickle.load(f)

    # <-- MODIFICATION: Group files by pair ID instead of triad ID.
    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("_ave_epo.fif")]
    
    pair_files = defaultdict(list)
    for f in all_files:
        pair_id = get_pair_from_filename(f)
        if pair_id:
            pair_files[pair_id].append(f)
        else:
            print(f"Warning: Could not find pair ID for file {f}")

    # Split by pair ID for train, val, test
    all_pair_ids = sorted(pair_files.keys())
    np.random.seed(42) # for reproducibility
    np.random.shuffle(all_pair_ids)

    # Splitting pairs: ~80% train, ~10% validation, ~10% test
    train_split = int(len(all_pair_ids) * 0.8)
    val_split = int(len(all_pair_ids) * 0.9)

    train_pair_ids = all_pair_ids[:train_split]
    val_pair_ids = all_pair_ids[train_split:val_split]
    test_pair_ids = all_pair_ids[val_split:]

    print(f"Total pairs: {len(all_pair_ids)}")
    print(f"Train pairs ({len(train_pair_ids)}): {train_pair_ids}")
    print(f"Validation pairs ({len(val_pair_ids)}): {val_pair_ids}")
    print(f"Test pairs ({len(test_pair_ids)}): {test_pair_ids}")

    # Create file lists based on pair splits
    train_files = [f for pid in train_pair_ids for f in pair_files[pid]]
    val_files = [f for pid in val_pair_ids for f in pair_files[pid]]
    test_files = [f for pid in test_pair_ids for f in pair_files[pid]]

    # Set your processed data path
    final_root = os.getenv("CIRCLING_PROCESSED_PATH", os.path.join(base_path, "processed_data"))
    
    # Create the train, val, test sample folders
    train_dump_folder = os.path.join(final_root, "train")
    val_dump_folder = os.path.join(final_root, "val")
    test_dump_folder = os.path.join(final_root, "test")
    
    os.makedirs(train_dump_folder, exist_ok=True)
    os.makedirs(val_dump_folder, exist_ok=True)
    os.makedirs(test_dump_folder, exist_ok=True)

    # Prepare parameters for the processing function
    parameters = []
    
    # <-- MODIFICATION: The parameters now include the bad epochs dictionary.
    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            filepath = os.path.join(raw_dataset_path, f)
            params.append([filepath, dump_folder, standard_channels, bad_short_epochs])
        return params

    parameters.extend(prepare_params(train_files, train_dump_folder))
    parameters.extend(prepare_params(val_files, val_dump_folder))
    parameters.extend(prepare_params(test_files, test_dump_folder))

    # Use a number of processes suitable for your machine
    with Pool(processes=8) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")