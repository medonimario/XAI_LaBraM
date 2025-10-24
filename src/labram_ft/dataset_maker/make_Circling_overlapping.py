import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import mne
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# Standard channel order remains the same.
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

# <-- MODIFICATION: The segmentation function is no longer needed, as the logic is moved directly
# into the main processing function for clarity and correctness.

# <-- MODIFICATION: The main processing function has been rewritten to handle filtering correctly.
def process_epoch_file(params):
    """
    Processes a single MNE epoch file. It loads data, and for each 18s trial,
    it generates clean 4s overlapping segments by checking against bad 6s time ranges
    that are local to that trial.
    """
    filepath, dump_folder, chOrder_standard, bad_epochs_dict = params
    
    base_filename = os.path.basename(filepath).split('_ave_epo.fif')[0]
    print("Processing", base_filename)

    try:
        epochs = mne.read_epochs(filepath, preload=True, verbose=False)
        
        pair_n_str = re.search(r'pair(\d+)', base_filename).group(1)
        practice_trial_cut = 12 if pair_n_str == '2030' else 6
        
        epochs = epochs.crop(tmin=2, verbose=False)[practice_trial_cut:]

        conditions_to_process = {'open': 1, 'closed1': 0}

        for condition, label in conditions_to_process.items():
            if condition not in epochs.event_id:
                continue

            # --- 1. Identify Bad Time Ranges PER TRIAL ---
            bad_epoch_key = f"{pair_n_str}_{condition}"
            bad_6s_indices_global = bad_epochs_dict.get(bad_epoch_key, [])
            print(f"Bad 6s segments for {bad_epoch_key}: {bad_6s_indices_global}")
            
            # This dictionary will map a trial index to its bad time ranges
            # e.g., { trial_5: [(0.0, 6.0)], trial_13: [(12.0, 18.0)] }
            bad_ranges_per_trial = defaultdict(list)
            for global_idx in bad_6s_indices_global:
                # An 18s epoch has three 6s segments
                trial_idx = global_idx // 3
                segment_in_trial = global_idx % 3
                bad_start = segment_in_trial * 6.0
                bad_end = bad_start + 6.0
                bad_ranges_per_trial[trial_idx].append((bad_start, bad_end))

            # --- 2. Segment and Filter using a Sliding Window on NumPy arrays ---
            sfreq = epochs.info['sfreq']
            all_trials_data = epochs[condition].get_data(units='uV')
            
            segm_len_samples = int(4 * sfreq)  # 4 seconds
            overlap_samples = int(2 * sfreq)   # 2 seconds
            step_size_samples = segm_len_samples - overlap_samples

            clean_segments_data = [] # This will store the raw data of all good segments

            # Loop over each 18-second trial
            for trial_idx, single_trial_data in enumerate(all_trials_data):
                bad_ranges_for_this_trial = bad_ranges_per_trial.get(trial_idx, [])
                n_samples_in_trial = single_trial_data.shape[1]
                
                # Apply sliding window to this trial
                for start_sample in range(0, n_samples_in_trial - segm_len_samples + 1, step_size_samples):
                    end_sample = start_sample + segm_len_samples
                    
                    # Get the time range of this 4s segment
                    tmin = start_sample / sfreq
                    tmax = end_sample / sfreq
                    
                    is_bad = False
                    for bad_start, bad_end in bad_ranges_for_this_trial:
                        # Check for overlap
                        if tmin < bad_end and tmax > bad_start:
                            print(f"Segment {tmin}-{tmax}s in trial {trial_idx} overlaps with bad range {bad_start}-{bad_end}s")
                            is_bad = True
                            break
                    
                    if not is_bad:
                        segment_data = single_trial_data[:, start_sample:end_sample]
                        clean_segments_data.append(segment_data)

            # --- 3. Process and Save Each Clean Segment ---
            for i, segment_data in enumerate(clean_segments_data):
                # Reshape to (1, n_channels, n_samples) to create a temporary MNE object
                temp_epoch_obj = mne.EpochsArray(segment_data[np.newaxis, :, :], epochs.info, verbose=False)

                if temp_epoch_obj.info['sfreq'] != 200:
                    temp_epoch_obj.resample(200, n_jobs=1, verbose=False)

                if chOrder_standard:
                    temp_epoch_obj.reorder_channels(chOrder_standard)
                
                final_epoch_data = temp_epoch_obj.get_data()[0]
                dump_path = os.path.join(
                    dump_folder, f"{base_filename}_condition_{condition}_segment{i}.pkl"
                )
                
                pickle.dump(
                    {"X": final_epoch_data, "y": label},
                    open(dump_path, "wb"),
                )

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        # Make sure to handle potential race conditions if writing from multiple processes
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(f"{filepath}\n")

# The rest of the script (get_pair_from_filename, __main__) can remain exactly the same.
def get_pair_from_filename(filename):
    match = re.search(r'pair(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    base_path = os.getenv("CIRCLING_DATASET_PATH")
    raw_dataset_path = os.path.join(base_path, "preprocessed/")
    bad_epochs_pickle_path = os.path.join(base_path, "dict_with_short_bad_epochs.pickle")

    print(f"Loading bad epochs dictionary from: {bad_epochs_pickle_path}")
    with open(bad_epochs_pickle_path, "rb") as f:
        bad_short_epochs = pickle.load(f)

    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("_ave_epo.fif")]
    
    pair_files = defaultdict(list)
    for f in all_files:
        pair_id = get_pair_from_filename(f)
        if pair_id:
            pair_files[pair_id].append(f)
        else:
            print(f"Warning: Could not find pair ID for file {f}")

    all_pair_ids = sorted(pair_files.keys())
    np.random.seed(42)
    np.random.shuffle(all_pair_ids)

    train_split = int(len(all_pair_ids) * 0.8)
    val_split = int(len(all_pair_ids) * 0.9)

    train_pair_ids = all_pair_ids[:train_split]
    val_pair_ids = all_pair_ids[train_split:val_split]
    test_pair_ids = all_pair_ids[val_split:]

    print(f"Total pairs: {len(all_pair_ids)}")
    print(f"Train pairs ({len(train_pair_ids)}): {train_pair_ids}")
    print(f"Validation pairs ({len(val_pair_ids)}): {val_pair_ids}")
    print(f"Test pairs ({len(test_pair_ids)}): {test_pair_ids}")

    train_files = [f for pid in train_pair_ids for f in pair_files[pid]]
    val_files = [f for pid in val_pair_ids for f in pair_files[pid]]
    test_files = [f for pid in test_pair_ids for f in pair_files[pid]]

    final_root = os.getenv("CIRCLING_PROCESSED_PATH", os.path.join(base_path, "processed_overlapping/"))
    
    train_dump_folder = os.path.join(final_root, "train")
    val_dump_folder = os.path.join(final_root, "val")
    test_dump_folder = os.path.join(final_root, "test")
    
    os.makedirs(train_dump_folder, exist_ok=True)
    os.makedirs(val_dump_folder, exist_ok=True)
    os.makedirs(test_dump_folder, exist_ok=True)

    parameters = []
    
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