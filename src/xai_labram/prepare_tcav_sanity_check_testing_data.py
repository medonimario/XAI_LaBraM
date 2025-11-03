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
    dataset_root = os.getenv("CIRCLING_DATASET_PATH")
    print(f"Using dataset root: {dataset_root}")
    processed_dataset_root = os.path.join(dataset_root, "processed_overlapping")

    output_dir = os.path.join(dataset_root, "tcav_sanity_check_data_test_split/") # MODIFIED: New output dir
    os.makedirs(output_dir, exist_ok=True)
    
    num_concept_examples = 100
    num_random_examples = 100
    
    # --- Define Labels ---
    CONCEPT_LABEL = 1  # 'Open' condition
    RANDOM_LABEL = 0   # 'Closed1' condition
    TARGET_CLASS_LABEL = 1 # We are explaining the 'Open' class

    # MODIFIED: We only need to scan the test directory now
    print("Scanning test dataset directory...")
    test_dir = os.path.join(processed_dataset_root, "test")
    test_files_by_label = get_files_and_labels(test_dir)

    # --- MODIFIED: Split the Test Set ---
    
    # Get all files for our labels from the test set
    all_concept_label_files = test_files_by_label.get(CONCEPT_LABEL, [])
    all_random_label_files = test_files_by_label.get(RANDOM_LABEL, [])

    # Shuffle them to ensure random splits
    random.seed(42)
    random.shuffle(all_concept_label_files)
    random.shuffle(all_random_label_files) # Good practice, though only one split is used here

    # Split the 'Open' (CONCEPT_LABEL) files into two halves
    # One half will be the *source pool* for the concept set (P_C)
    # The other half will be the *target class* set (X_k)
    concept_split_point = len(all_concept_label_files) // 2
    concept_source_pool = all_concept_label_files[:concept_split_point]
    target_class_set = all_concept_label_files[concept_split_point:] # MODIFIED: This is now our target set
    
    # For the 'Closed1' (RANDOM_LABEL) files, we can just use all of them as the source pool
    # since we aren't using 'Closed1' as a target class.
    random_source_pool = all_random_label_files

    print(f"Total 'Open' (Label {CONCEPT_LABEL}) files in test: {len(all_concept_label_files)}")
    print(f" -> Using {len(concept_source_pool)} as source pool for concept examples.")
    print(f" -> Using {len(target_class_set)} as target class examples.")
    print(f"Total 'Closed1' (Label {RANDOM_LABEL}) files in test: {len(all_random_label_files)}")
    print(f" -> Using {len(random_source_pool)} as source pool for random examples.")


    # --- 1. Create Concept Dataset (P_C) ---
    # MODIFIED: Sample from the concept_source_pool (first half of test set)
    if len(concept_source_pool) < num_concept_examples:
        print(f"Warning: Not enough examples for concept '{CONCEPT_LABEL}'. Found {len(concept_source_pool)}, needed {num_concept_examples}.")
        num_concept_examples = len(concept_source_pool)
        
    concept_set = random.sample(concept_source_pool, num_concept_examples)
    print(f"Selected {len(concept_set)} files for the 'Open' concept dataset.")

    # --- 2. Create Random/Negative Dataset (N) ---
    # MODIFIED: Sample from the random_source_pool (all 'Closed1' test files)
    if len(random_source_pool) < num_random_examples:
        print(f"Warning: Not enough examples for random set '{RANDOM_LABEL}'. Found {len(random_source_pool)}, needed {num_random_examples}.")
        num_random_examples = len(random_source_pool)

    random_set = random.sample(random_source_pool, num_random_examples)
    print(f"Selected {len(random_set)} files for the 'Closed1' random dataset.")

    # --- 3. Create Target Class Dataset (X_k) ---
    # MODIFIED: This is already defined as the second half of the split
    print(f"Using {len(target_class_set)} files from the test set split for the 'Open' target class dataset.")

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