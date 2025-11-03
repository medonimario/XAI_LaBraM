import os
import pickle
import random
import json
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

def get_files_and_labels(directory):
    """Scans a directory, loads pickle files, and returns a dict of label to file paths."""
    label_to_files = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    label = data['y']
                    label_to_files[label].append(filepath)
            except Exception as e:
                print(f"Could not process {filepath}: {e}")
    return label_to_files

def main():
    # --- Configuration ---
    # Path to the processed Circling dataset (containing train/, val/, test/ folders)
    dataset_root = os.getenv("CIRCLING_DATASET_PATH")
    print(f"Using dataset root: {dataset_root}")
    processed_dataset_root = os.path.join(dataset_root, "processed_overlapping")

    # Where to save the file lists for TCAV
    output_dir = os.path.join(dataset_root, "tcav_sanity_check_data/")
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of examples for the concept and random sets
    num_concept_examples = 100
    num_random_examples = 100
    
    # --- Define Labels ---
    # From make_Circling_overlapping.py: {'open': 1, 'closed1': 0}
    CONCEPT_LABEL = 1  # 'Open' condition
    RANDOM_LABEL = 0   # 'Closed1' condition
    TARGET_CLASS_LABEL = 1 # We are explaining the 'Open' class

    print("Scanning dataset directories...")
    train_dir = os.path.join(processed_dataset_root, "train")
    test_dir = os.path.join(processed_dataset_root, "test")

    train_files_by_label = get_files_and_labels(train_dir)
    test_files_by_label = get_files_and_labels(test_dir)

    # --- 1. Create Concept Dataset (P_C) ---
    concept_files = train_files_by_label.get(CONCEPT_LABEL, [])
    if len(concept_files) < num_concept_examples:
        print(f"Warning: Not enough examples for concept '{CONCEPT_LABEL}'. Found {len(concept_files)}, needed {num_concept_examples}.")
        num_concept_examples = len(concept_files)
        
    random.seed(42)
    concept_set = random.sample(concept_files, num_concept_examples)
    print(f"Selected {len(concept_set)} files for the 'Open' concept dataset.")

    # --- 2. Create Random/Negative Dataset (N) ---
    random_files = train_files_by_label.get(RANDOM_LABEL, [])
    if len(random_files) < num_random_examples:
        print(f"Warning: Not enough examples for random set '{RANDOM_LABEL}'. Found {len(random_files)}, needed {num_random_examples}.")
        num_random_examples = len(random_files)

    random_set = random.sample(random_files, num_random_examples)
    print(f"Selected {len(random_set)} files for the 'Closed1' random dataset.")

    # --- 3. Create Target Class Dataset (X_k) ---
    target_class_set = test_files_by_label.get(TARGET_CLASS_LABEL, [])
    print(f"Selected all {len(target_class_set)} files from the test set for the 'Open' target class dataset.")

    # --- Save the file lists ---
    output_paths = {
        "concept_files": os.path.join(output_dir, "concept_open.json"),
        "random_files": os.path.join(output_dir, "random_closed1.json"),
        "target_files": os.path.join(output_dir, "target_open.json"),
    }

    with open(output_paths["concept_files"], "w") as f:
        json.dump(concept_set, f, indent=4)
    with open(output_paths["random_files"], "w") as f:
        json.dump(random_set, f, indent=4)
    with open(output_paths["target_files"], "w") as f:
        json.dump(target_class_set, f, indent=4)

    print(f"\nSuccessfully created data manifests for TCAV in the '{output_dir}' directory.")
    print("You are now ready for Step 4: Model Adaptation and Activation Extraction.")

if __name__ == "__main__":
    main()