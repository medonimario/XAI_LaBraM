import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import mne
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import shutil
from collections import defaultdict

# It's good practice to keep a standard channel order for model compatibility.
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
    Processes a single MNE epoch file.
    - Takes ALL epochs available in the file (no condition filtering).
    - Uses the preassigned subject-level label (based on Exp_id -> Friend_status).
    - Saves each epoch individually with keys {"X", "y"} where y is 0/1.
    """
    filepath, dump_folder, chOrder_standard, subject_label, exp_id = params

    print("Processing", os.path.basename(filepath), f"(Exp_id={exp_id}, label={subject_label})")
    try:
        # Load the pre-epoched data file
        epochs = mne.read_epochs(filepath, preload=True, verbose=False)

        # Optional: crop, resample, reorder (kept as in your script)
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
            # Keep condition name in the filename for traceability (label does NOT depend on it)
            event_code = epochs.events[i, 2]
            condition_name = rev_event_id.get(event_code, "UNK")

            epoch_data = data[i, :, :]

            dump_path = os.path.join(
                dump_folder, f"{base_filename}_cond-{condition_name}_epoch{i}.pkl"
            )

            # Subject-level label (0 = 'No', 1 = 'Yes')
            pickle.dump({"X": epoch_data, "y": subject_label}, open(dump_path, "wb"))

    except Exception as e:
        print(f"ERROR processing file {filepath}: {e}")
        with open("my-dataset-process-error-files.txt", "a") as f:
            f.write(filepath + "\n")

def get_triad_from_filename(filename):
    """
    Extracts the triad ID from a filename like '301A_FG_...-epo.fif'.
    It captures the number at the start of the string.
    """
    match = re.search(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def get_expid_from_filename(filename):
    """
    Extracts Exp_id (e.g., '301A') from filenames like '301A_FG_preprocessed-epo.fif'.
    """
    match = re.match(r'^(\d+[A-Za-z])', filename)
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    # --- Load subject-level labels from your dataframe -----------------------
    # Adjust this path to wherever the pickle lives in your environment.
    # Example uses an env var; you can hardcode if you prefer.
    overview_pkl_path = os.getenv("FG_DATASET_PATH") + "FG_overview_df_v2.pkl"
    df = pd.read_pickle(overview_pkl_path)
    # Map Exp_id -> 0/1 (No/Yes)
    expid_to_label = {
        str(row.Exp_id): (1 if str(row.Friend_status).strip().lower() == "yes" else 0)
        for _, row in df[['Exp_id', 'Friend_status']].iterrows()
    }

    # --- Paths and file discovery -------------------------------------------
    raw_dataset_path = os.getenv("FG_DATASET_PATH") + "preprocessed/"
    all_files = [f for f in os.listdir(raw_dataset_path) if f.endswith("-epo.fif")]

    # Group files by triad for split (unchanged)
    triad_files = defaultdict(list)
    for f in all_files:
        triad_id = get_triad_from_filename(f)
        if triad_id:
            triad_files[triad_id].append(f)
        else:
            print(f"Warning: Could not find triad ID for file {f}")

    # Triad-level split (unchanged)
    all_triad_ids = sorted(triad_files.keys())
    np.random.seed(42)
    np.random.shuffle(all_triad_ids)

    train_split = int(len(all_triad_ids) * 0.8)
    val_split = int(len(all_triad_ids) * 0.9)

    train_triad_ids = all_triad_ids[:train_split]
    val_triad_ids   = all_triad_ids[train_split:val_split]
    test_triad_ids  = all_triad_ids[val_split:]

    print(f"Total triads: {len(all_triad_ids)}")
    print(f"Train triads ({len(train_triad_ids)}): {train_triad_ids}")
    print(f"Validation triads ({len(val_triad_ids)}): {val_triad_ids}")
    print(f"Test triads ({len(test_triad_ids)}): {test_triad_ids}")

    # Build file lists for each split (unchanged)
    train_files = [f for tid in train_triad_ids for f in triad_files[tid]]
    val_files   = [f for tid in val_triad_ids   for f in triad_files[tid]]
    test_files  = [f for tid in test_triad_ids  for f in triad_files[tid]]

    # Output roots
    final_root = os.getenv("FG_DATASET_PATH")
    train_dump_folder = os.path.join(final_root, "processed_friendship", "train")
    val_dump_folder   = os.path.join(final_root, "processed_friendship", "val")
    test_dump_folder  = os.path.join(final_root, "processed_friendship", "test")

    os.makedirs(train_dump_folder, exist_ok=True)
    os.makedirs(val_dump_folder,   exist_ok=True)
    os.makedirs(test_dump_folder,  exist_ok=True)

    # Prepare parameters for the processing function
    parameters = []

    def prepare_params(file_list, dump_folder):
        params = []
        for f in file_list:
            exp_id = get_expid_from_filename(f)
            if exp_id is None:
                print(f"Warning: Could not parse Exp_id from {f}; skipping.")
                continue
            if exp_id not in expid_to_label:
                print(f"Warning: Exp_id {exp_id} not found in overview dataframe; skipping {f}.")
                continue
            label = expid_to_label[exp_id]  # 0 for 'No', 1 for 'Yes'
            filepath = os.path.join(raw_dataset_path, f)
            params.append([filepath, dump_folder, standard_channels, label, exp_id])
        return params

    parameters.extend(prepare_params(train_files, train_dump_folder))
    parameters.extend(prepare_params(val_files,   val_dump_folder))
    parameters.extend(prepare_params(test_files,  test_dump_folder))

    # Multiprocessing
    with Pool(processes=8) as pool:
        pool.map(process_epoch_file, parameters)

    print("Processing complete.")