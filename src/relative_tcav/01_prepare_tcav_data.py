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
            if filename.endswith(".pkl") and not filename.startswith("._"):
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
            if filename.endswith(".pkl") and not filename.startswith("._"):
                file_paths.append(os.path.join(directory, filename))
    
    logging.info(f"Found {len(file_paths)} total files.")
    return file_paths

def main(args):
    """
    Main function to prepare data manifests for RELATIVE TCAV.
    """
    # 1. Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info("Starting Relative TCAV data preparation script.")
    logging.info(f"Script arguments:\n{json.dumps(vars(args), indent=2)}")
    random.seed(args.seed)

    concept_source_pool = []
    contrast_source_pool = []  # --- MODIFIED: Renamed from random_source_pool
    target_class_set = []
    concept_set = []
    contrast_set = []          # --- MODIFIED: Single contrast set instead of list of random sets

    # 2. Load Data Pools based on Mode (Sanity Check vs. Standard)
    if args.sanity_check:
        # --- SANITY CHECK MODE ---
        if args.skip_target_set or args.skip_contrast_set: # --- MODIFIED: Updated flag name
            logging.warning("Ignoring skip flags in --sanity_check mode.")

        # In Relative Sanity Check:
        # Split target_dir into Concept A (Half) and Contrast B (Half) to test "Same vs Same".
        # Or, if contrast_label is provided, use that.
        # Ideally for "Sanity Check", we want A and B to be statistically identical to check if TCAV -> 0 (Random).
        
        logging.info("--- Running in SANITY CHECK Mode ---")
        
        # Load all files for the target label
        all_target_files = load_labeled_files([args.target_dir], args.target_label)

        # Shuffle and split
        random.shuffle(all_target_files)
        # We need 3 splits: Target Set (for gradients), Concept A, Contrast B
        # Let's do 50% Target, 25% Concept A, 25% Contrast B
        
        n_total = len(all_target_files)
        if n_total < 10:
             raise ValueError("Not enough target files for sanity check splitting.")

        split_1 = int(n_total * 0.5)
        split_2 = int(n_total * 0.75)

        target_class_set = all_target_files[:split_1]
        concept_source_pool = all_target_files[split_1:split_2]
        contrast_source_pool = all_target_files[split_2:]
        
        logging.info(f"Loaded {n_total} target files from {args.target_dir} and split into:")
        logging.info(f"  {len(target_class_set)} files for Target Class set (Gradients)")
        logging.info(f"  {len(concept_source_pool)} files for Concept A pool")
        logging.info(f"  {len(contrast_source_pool)} files for Contrast B pool")

    else:
        # --- STANDARD MODE ---
        logging.info("--- Running in STANDARD Mode ---")

        # 1. Load Target Class Set (X_k)
        if not args.skip_target_set:
            logging.info("Loading Target Class set (X_k)...")
            target_class_set = load_labeled_files([args.target_dir], args.target_label)
            if not target_class_set:
                logging.warning(f"No target class files found for label {args.target_label} in {args.target_dir}")
        else:
            logging.info("Skipping Target Class set loading.")

        # 2. Load Concept Source Pool (P_C)
        logging.info("Loading Concept source pool (P_C)...")
        if args.concept_label is not None:
            logging.info(f"Using Labeled Concept mode (label={args.concept_label}).")
            concept_source_pool = load_labeled_files(args.concept_dirs, args.concept_label)
        else:
            logging.info("Using Unlabeled Concept mode (all files in dir).")
            concept_source_pool = load_unlabeled_files(args.concept_dirs)
        
        if not concept_source_pool:
            raise ValueError("Concept source pool is empty.")

        # 3. Load Contrast Source Pool (P_Contrast) --- MODIFIED SECTION ---
        if not args.skip_contrast_set:
            logging.info("Loading Contrast source pool (P_Contrast)...")
            if args.contrast_label is not None:
                logging.info(f"Using Labeled Contrast mode (label={args.contrast_label}).")
                contrast_source_pool = load_labeled_files(args.contrast_dirs, args.contrast_label)
            else:
                logging.info("Using Unlabeled Contrast mode (all files in dir).")
                contrast_source_pool = load_unlabeled_files(args.contrast_dirs)
                
            if not contrast_source_pool:
                raise ValueError("Contrast source pool is empty.")
        else:
            logging.info("Skipping Contrast source pool loading.")


    # 3. Create Final Datasets (Sampling)
    
    # --- Create Concept Set (P_C) ---
    if concept_source_pool:
        num_concept = min(len(concept_source_pool), args.num_examples) # Renamed arg
        concept_set = random.sample(concept_source_pool, num_concept)
        logging.info(f"Created Concept set with {len(concept_set)} files.")
    
    # --- Create Contrast Set (P_Contrast) --- MODIFIED SECTION ---
    # We no longer loop for N runs. We just create one set.
    if not args.skip_contrast_set or args.sanity_check:
        if contrast_source_pool:
            num_contrast = min(len(contrast_source_pool), args.num_examples)
            contrast_set = random.sample(contrast_source_pool, num_contrast)
            logging.info(f"Created Contrast set with {len(contrast_set)} files.")
        else:
            logging.warning("Contrast pool empty, no set created.")
    else:
        logging.info("Skipping Contrast set creation.")

    
    # 4. Save Manifests
    logging.info("Saving manifest files...")
    
    paths = {
        "concept_set": os.path.join(args.output_dir, "concept_set.json"),
        "contrast_set": os.path.join(args.output_dir, "contrast_set.json"), # --- MODIFIED filename
        "target_class_set": os.path.join(args.output_dir, "target_class_set.json"),
        "summary": os.path.join(args.output_dir, "_summary.json")
    }
    
    summary = {
        "info": "Relative TCAV Data Manifests",
        "config": vars(args),
        "counts": {
            "concept_set_size": len(concept_set),
            "contrast_set_size": len(contrast_set), # --- MODIFIED
            "target_class_set_size": len(target_class_set)
        }
    }
    
    try:
        with open(paths["concept_set"], 'w') as f:
            json.dump(concept_set, f, indent=4)
        
        # --- MODIFIED: Write contrast set
        if (not args.skip_contrast_set or args.sanity_check) and contrast_set:
            with open(paths["contrast_set"], 'w') as f:
                json.dump(contrast_set, f, indent=4)
        
        if (not args.skip_target_set or args.sanity_check) and target_class_set:
            with open(paths["target_class_set"], 'w') as f:
                json.dump(target_class_set, f, indent=4)
            
        with open(paths["summary"], 'w') as f:
            json.dump(summary, f, indent=4)
            
    except Exception as e:
        logging.error(f"Failed to write manifest files: {e}")
        raise
        
    logging.info("--- Data preparation complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data manifests for Relative TCAV.")

    # --- General ---
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output JSON manifest files.")
    parser.add_argument("--num_examples", type=int, default=104, # Renamed from per_run
                        help="Number of examples to sample for concept and contrast sets.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # --- Target Class (X_k) ---
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Directory containing the target class data.")
    parser.add_argument("--target_label", type=int, required=True,
                        help="The numerical label for the target class.")

    # --- Concept (A) ---
    parser.add_argument("--concept_dirs", type=str, nargs='+',
                        help="Directories for Concept A.")
    parser.add_argument("--concept_label", type=int, default=None,
                        help="Optional: Label filter for Concept A.")

    # --- Contrast (B) --- MODIFIED ARGS ---
    parser.add_argument("--contrast_dirs", type=str, nargs='+',
                        help="Directories for Contrast Concept B.")
    parser.add_argument("--contrast_label", type=int, default=None,
                        help="Optional: Label filter for Contrast B.")

    # --- Mode Toggle ---
    parser.add_argument("--sanity_check", action='store_true',
                        help="Enable sanity check mode (Splits target_dir into A, B, and Target).")
    
    # --- Skip Flags ---
    parser.add_argument("--skip_target_set", action='store_true')
    parser.add_argument("--skip_contrast_set", action='store_true') # --- MODIFIED
    
    args = parser.parse_args()
    
    if not args.sanity_check:
        if not args.concept_dirs:
            parser.error("In Standard Mode, --concept_dirs is required.")
        if not args.contrast_dirs and not args.skip_contrast_set: # --- MODIFIED
            parser.error("In Standard Mode, --contrast_dirs is required unless --skip_contrast_set is used.")

    main(args)