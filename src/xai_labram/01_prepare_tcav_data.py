import os
import pickle
import random
import json
import argparse
import logging
from collections import defaultdict

def setup_logging(output_dir):
    """Configures logging to file and console."""
    log_file = os.path.join(output_dir, "_data_preparation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")

def load_labeled_files(directories, required_label):
    """
    Scans directories for .pkl files and returns paths where 
    data['y'] matches the required_label.
    """
    file_paths = []
    if not isinstance(directories, list):
        directories = [directories]
        
    for directory in directories:
        if not os.path.isdir(directory):
            logging.warning(f"Directory not found, skipping: {directory}")
            continue
        
        logging.info(f"Scanning {directory} for label '{required_label}'...")
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "rb") as f:
                        data = pickle.load(f)
                    
                    if 'y' in data and data['y'] == required_label:
                        file_paths.append(filepath)
                except Exception as e:
                    logging.warning(f"Could not process {filepath}: {e}")
    
    logging.info(f"Found {len(file_paths)} files matching label '{required_label}'.")
    return file_paths

def load_unlabeled_files(directories):
    """Scans directories for .pkl files and returns all paths."""
    file_paths = []
    if not isinstance(directories, list):
        directories = [directories]

    for directory in directories:
        if not os.path.isdir(directory):
            logging.warning(f"Directory not found, skipping: {directory}")
            continue

        logging.info(f"Scanning {directory} for all .pkl files...")
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                file_paths.append(os.path.join(directory, filename))
    
    logging.info(f"Found {len(file_paths)} total files.")
    return file_paths

def main(args):
    """
    Main function to prepare data manifests for TCAV.
    """
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info("Starting TCAV data preparation script.")
    logging.info(f"Script arguments:\n{json.dumps(vars(args), indent=2)}")
    random.seed(args.seed)

    concept_source_pool = []
    random_source_pool = []
    target_class_set = []

    # 2. Load Data Pools based on Mode (Sanity Check vs. Standard)
    if args.sanity_check:
        # --- SANITY CHECK MODE ---
        # In this mode, we split the target_dir test set.
        # Half of target_label files -> Concept Pool
        # Other half of target_label files -> Target Set
        # All of negative_label files -> Random/Negative Pool
        logging.info("--- Running in SANITY CHECK Mode ---")
        if args.negative_label is None:
            logging.error("Sanity check mode requires --negative_label.")
            raise ValueError("--negative_label is required for sanity check mode.")

        # Load all files for both labels from the *single* target directory
        all_target_files = load_labeled_files([args.target_dir], args.target_label)
        all_negative_files = load_labeled_files([args.target_dir], args.negative_label)

        # Shuffle and split the target class files
        random.shuffle(all_target_files)
        split_point = len(all_target_files) // 2
        if split_point == 0 or len(all_target_files) < 2:
            logging.error(f"Not enough target files ({len(all_target_files)}) to split for sanity check.")
            raise ValueError("Not enough target files to split.")

        concept_source_pool = all_target_files[:split_point]
        target_class_set = all_target_files[split_point:]
        
        # The negative files from this dataset are our "random" pool
        random_source_pool = all_negative_files
        
        logging.info(f"Loaded {len(all_target_files)} target files from {args.target_dir} and split into:")
        logging.info(f"  {len(concept_source_pool)} files for Concept pool")
        logging.info(f"  {len(target_class_set)} files for Target Class set")
        logging.info(f"Loaded {len(random_source_pool)} negative files from {args.target_dir} for Random/Negative pool")

    else:
        # --- STANDARD MODE ---
        # In this mode, datasets are treated as independent.
        # Target set comes from target_dir.
        # Concept set comes from concept_dirs.
        # Random/Negative set comes from random_dirs.
        logging.info("--- Running in STANDARD Mode ---")

        # 1. Load Target Class Set (X_k)
        logging.info("Loading Target Class set (X_k)...")
        target_class_set = load_labeled_files([args.target_dir], args.target_label)
        if not target_class_set:
            logging.warning(f"No target class files found for label {args.target_label} in {args.target_dir}")

        # 2. Load Concept Source Pool (P_C)
        logging.info("Loading Concept source pool (P_C)...")
        if args.concept_label is not None:
            logging.info(f"Using Labeled Concept mode (label={args.concept_label}).")
            concept_source_pool = load_labeled_files(args.concept_dirs, args.concept_label)
        else:
            logging.info("Using Unlabeled Concept mode (all files in dir).")
            concept_source_pool = load_unlabeled_files(args.concept_dirs)
        
        if not concept_source_pool:
            logging.error(f"No concept files found in {args.concept_dirs} (Label: {args.concept_label}).")
            raise ValueError("Concept source pool is empty.")

        # 3. Load Random/Negative Source Pool (N)
        logging.info("Loading Random/Negative source pool (N)...")
        if args.negative_label is not None:
            logging.info(f"Using 'Negative' mode (label={args.negative_label}).")
            random_source_pool = load_labeled_files(args.random_dirs, args.negative_label)
        else:
            logging.info("Using 'Random' mode (all files in dir).")
            random_source_pool = load_unlabeled_files(args.random_dirs)
            
        if not random_source_pool:
            logging.error(f"No random/negative files found in {args.random_dirs} (Label: {args.negative_label}).")
            raise ValueError("Random/Negative source pool is empty.")

    # 3. Create Final Datasets
    
    # --- Create Concept Set (P_C) ---
    num_concept = min(len(concept_source_pool), args.num_examples_per_run)
    if num_concept < args.num_examples_per_run:
        logging.warning(f"Only found {num_concept} concept examples, requested {args.num_examples_per_run}.")
    
    concept_set = random.sample(concept_source_pool, num_concept)
    logging.info(f"Created Concept set with {len(concept_set)} files.")
    
    # --- Create Random/Negative Sets (N_runs) ---
    random_sets = []
    num_random_per_run = min(len(random_source_pool), args.num_examples_per_run)
    if num_random_per_run < args.num_examples_per_run:
        logging.warning(f"Only {num_random_per_run} random/negative examples available per run, requested {args.num_examples_per_run}.")
    
    # Check if we have enough *unique* samples for all runs without replacement
    if len(random_source_pool) < args.num_examples_per_run * args.num_runs:
        logging.warning(f"Total random pool ({len(random_source_pool)}) is smaller than "
                        f"total samples needed ({args.num_examples_per_run * args.num_runs}). "
                        "Sets will be sampled *with* replacement (i.e., files may be reused across sets).")
    
    logging.info(f"Creating {args.num_runs} random/negative sets of size {num_random_per_run}...")
    for _ in range(args.num_runs):
        random_sets.append(random.sample(random_source_pool, num_random_per_run))
    
    logging.info(f"Created {len(random_sets)} random/negative sets.")
    
    # Target Class set (X_k) is already prepared as target_class_set
    logging.info(f"Using {len(target_class_set)} files for the Target Class set.")

    # 4. Save Manifests
    logging.info("Saving manifest files...")
    
    # Define paths
    paths = {
        "concept_set": os.path.join(args.output_dir, "concept_set.json"),
        "random_sets": os.path.join(args.output_dir, "random_sets.json"),
        "target_class_set": os.path.join(args.output_dir, "target_class_set.json"),
        "summary": os.path.join(args.output_dir, "_summary.json")
    }
    
    # Create summary dictionary
    summary = {
        "info": "TCAV Data Manifests",
        "config": vars(args),
        "counts": {
            "concept_set_size": len(concept_set),
            "num_random_sets": len(random_sets),
            "random_set_size": len(random_sets[0]) if random_sets else 0,
            "target_class_set_size": len(target_class_set)
        },
        "source_pool_counts": {
            "concept_source_pool": len(concept_source_pool),
            "random_source_pool": len(random_source_pool)
        }
    }
    
    # Write files
    try:
        with open(paths["concept_set"], 'w') as f:
            json.dump(concept_set, f, indent=4)
            
        with open(paths["random_sets"], 'w') as f:
            json.dump(random_sets, f, indent=4)
            
        with open(paths["target_class_set"], 'w') as f:
            json.dump(target_class_set, f, indent=4)
            
        with open(paths["summary"], 'w') as f:
            json.dump(summary, f, indent=4)
            
    except Exception as e:
        logging.error(f"Failed to write manifest files: {e}")
        raise
        
    logging.info("--- Data preparation complete. ---")
    logging.info(f"Concept set: {paths['concept_set']}")
    logging.info(f"Random sets: {paths['random_sets']}")
    logging.info(f"Target set: {paths['target_class_set']}")
    logging.info(f"Summary: {paths['summary']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data manifests for TCAV analysis.")

    # --- General ---
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output JSON manifest files and log.")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of random sets to generate for statistical testing.")
    parser.add_argument("--num_examples_per_run", type=int, default=100,
                        help="Number of examples to sample for the concept set and each random set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # --- Target Class (X_k) ---
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Directory containing the target class data (e.g., finetuned test set).")
    parser.add_argument("--target_label", type=int, required=True,
                        help="The numerical label for the target class (e.g., 1 for 'open').")

    # --- Concept (P_C) ---
    parser.add_argument("--concept_dirs", type=str, nargs='+',
                        help="One or more directories to source concept examples from.")
    parser.add_argument("--concept_label", type=int, default=None,
                        help="Optional: Label to filter for in concept_dirs. If not provided, all .pkl files are used.")

    # --- Random/Negative (N) ---
    parser.add_argument("--random_dirs", type=str, nargs='+',
                        help="One or more directories to source random/negative examples from.")
    parser.add_argument("--negative_label", type=int, default=None,
                        help="Optional: Label for 'negative' examples. If provided, filters random_dirs by this label. "
                             "If not, uses all files as 'random' examples.")

    # --- Mode Toggle ---
    parser.add_argument("--sanity_check", action='store_true',
                        help="Enable sanity check mode. In this mode, --target_dir is split to create all datasets. "
                             "--concept_dirs and --random_dirs are ignored. --negative_label *is* required.")
    
    args = parser.parse_args()
    
    # --- Argument Validation ---
    if not args.sanity_check and (not args.concept_dirs or not args.random_dirs):
        parser.error("In Standard Mode (default), --concept_dirs and --random_dirs are required.")
    
    if args.sanity_check and (args.concept_dirs or args.random_dirs):
        print("Warning: --concept_dirs and --random_dirs are ignored in --sanity_check mode.")

    main(args)