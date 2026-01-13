import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import warnings
from logger import Logger  # Assuming you have a logger.py file with the Logger class

class ConceptLabeling_Pipeline():
    def __init__(self, pair_n, input_path, save_path, condition=None, testing=False):
        self.pair_n = str(pair_n)
        self.input_path = input_path  # .../Epochs
        self.save_path = save_path    # .../Epochs_labeled
        self.condition = condition    # E.g., 'closed_baseline_epoch'
        self.testing = testing

        # Define file paths
        self.epochs_a_path = os.path.join(self.input_path, f"pair{self.pair_n}_a-epo.fif")
        self.epochs_b_path = os.path.join(self.input_path, f"pair{self.pair_n}_b-epo.fif")
        
        self.save_a_path = os.path.join(self.save_path, f"pair{self.pair_n}_a-epo.fif")
        self.save_b_path = os.path.join(self.save_path, f"pair{self.pair_n}_b-epo.fif")

        # Define frequency bands
        self.bands = {
            'theta': (4.0, 8.0),
            'alpha': (8.0, 12.0),
            'beta': (15.0, 25.0)
        }
        
        # Initialize the logger
        log_base_path = os.path.join(self.save_path, 'logs')
        self.logger = Logger(f"pair_{self.pair_n}_04_labeling", log_base_path)
        self.logger.log_text(f"Input folder: {self.input_path}")
        self.logger.log_text(f"Save folder: {self.save_path}")
        self.logger.log_text(f"Condition selected: {self.condition}")

        # Create the save directory
        os.makedirs(self.save_path, exist_ok=True)

    def process_subject(self, file_path, save_path, subject_label):
        """
        Loads, filters, labels, and saves epochs for a single subject.
        """
        self.logger.log_section(f"Processing Subject {subject_label.upper()} from {file_path}")
        
        # 1. Load Data
        try:
            epochs = mne.read_epochs(file_path, preload=True)
            self.logger.log_text(f"Loaded {len(epochs)} epochs.")
        except FileNotFoundError:
            self.logger.log_text(f"ERROR: File not found at {file_path}. Skipping.")
            print(f"ERROR: File not found for pair {self.pair_n}, subject {subject_label}.")
            return
        except Exception as e:
            self.logger.log_text(f"ERROR: Could not load file {file_path}. Reason: {e}")
            return
            
        if epochs.metadata is None:
            self.logger.log_text("ERROR: No metadata found. Cannot filter bad epochs. Skipping.")
            return
            
        metadata = epochs.metadata.copy()

        # 2. Filter out bad epochs
        if 'autoreject_bad' in metadata.columns:
            metadata['autoreject_bad'] = metadata['autoreject_bad'].fillna(False) # Treat NaN as 'not bad'
            n_before = len(epochs)
            epochs = epochs[metadata['autoreject_bad'] == False]
            n_after = len(epochs)
            self.logger.log_text(f"Filtered bad epochs: {n_before} -> {n_after} (dropped {n_before - n_after})")
        else:
            self.logger.log_text("WARNING: 'autoreject_bad' column not found. Proceeding with all epochs.")

        # 3. Filter by condition
        if self.condition:
            self.logger.log_text(f"Selecting only '{self.condition}' epochs.")
            n_before = len(epochs)
            try:
                epochs = epochs[self.condition]
                n_after = len(epochs)
                self.logger.log_text(f"Filtered by condition: {n_before} -> {n_after} (dropped {n_before - n_after})")
            except KeyError:
                self.logger.log_text(f"ERROR: Condition '{self.condition}' not found in epoch events. Skipping subject.")
                return

        # 4. Check if any epochs remain
        if len(epochs) == 0:
            self.logger.log_text("No epochs remaining after filtering. Skipping subject.")
            print(f"Pair {self.pair_n} subject {subject_label} has 0 epochs after filtering.")
            return
            
        self.logger.log_text(f"Final number of epochs to process: {len(epochs)}")

        # 5. Compute PSD
        # Suppress PSD warnings about windowing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
            psd = epochs_csd.compute_psd(method="welch", fmin=1.0, fmax=40.0, picks='csd')
            # psd = epochs.compute_psd(method="welch", fmin=1.0, fmax=40.0, picks='eeg')
        
        # Get the filtered metadata
        metadata = epochs.metadata.copy()

        # 6. Average Power Within Bands
        self.logger.log_subsection("Calculating and Normalizing Power")
        for band_name, (fmin, fmax) in self.bands.items():
            
            # Get power and average over frequencies
            band_power = psd.get_data(fmin=fmin, fmax=fmax).mean(axis=2)
            
            # Add to metadata, one column per channel
            for i, ch_name in enumerate(psd.ch_names):
                col_name = f"{band_name}_{ch_name}"
                metadata[col_name] = band_power[:, i]
        
        self.logger.log_text(f"Calculated raw power for {self.bands.keys()}.")

        # 7. Normalize Power (Z-Score)
        power_columns = [col for col in metadata.columns if any(col.startswith(band) for band in self.bands)]
        
        for col in power_columns:
            z_col_name = f"z_{col}"
            # Calculate z-score: (value - mean) / std_dev
            metadata[z_col_name] = zscore(metadata[col])
            
        self.logger.log_text("Z-scored all power values.")

        # Create "Spatial Specificity" columns
        for band in self.bands:
            # Get all z-score columns for this band
            z_cols = [c for c in metadata.columns if c.startswith(f"z_{band}_")]
            
            # Calculate mean Z-score across the scalp for this epoch (Global Power)
            metadata[f"global_z_{band}"] = metadata[z_cols].mean(axis=1)
            
            # Calculate Specificity: Channel Z - Global Z
            for z_col in z_cols:
                chan_name = z_col.split('_')[-1] # extract 'C3' from 'z_alpha_C3'
                spec_col = f"spec_{band}_{chan_name}"
                metadata[spec_col] = metadata[z_col] - metadata[f"global_z_{band}"]

        # 8. Save updated epochs
        epochs.metadata = metadata
        epochs.save(save_path, overwrite=True)
        self.logger.log_section(f"Successfully saved labeled epochs to {save_path}")
        print(f"  > Saved labeled epochs for subject {subject_label}")


    def run(self):
        print("\n\n")
        print("#"*50)
        print(f"SCRIPT 4: LABELING EPOCHS FOR PAIR {self.pair_n}")
        print("#"*50)

        try:
            # --- Process Subject A ---
            self.process_subject(self.epochs_a_path, self.save_a_path, 'a')

            # --- Process Subject B ---
            self.process_subject(self.epochs_b_path, self.save_b_path, 'b')

            print(f"\nLabeling script completed successfully for Pair {self.pair_n}!")
            self.logger.log_section("Labeling completed successfully!")

        except Exception as e:
            print(f"\nCRITICAL ERROR for Pair {self.pair_n}: {e}")
            self.logger.log_text(f"\n\nCRITICAL ERROR: {e}\n\n")
        
        finally:
            # Always convert the log to HTML
            self.logger.convert_markdown_to_html()


if __name__ == "__main__":
    # This is the list from your 02_circling_preprocess_epoch.py script
    all_pairs = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200,
                 2210, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370, 2380, 2390,
                 2410, 2420, 2430, 2440, 2450, 2460, 2470, 2480, 2490, 2500, 2510, 2520, 2530]
    
    # --- Configuration ---
    
    # This is the 'save_folder' from script 03
    input_folder = "/Volumes/T9/Circling_dataset/Epochs"
    
    # This is the new folder for our labeled output
    save_folder = "/Volumes/T9/Circling_dataset/Epochs_labeled"
    
    # Set to the condition you want to process.
    # Set to None to process ALL clean epochs.
    CONDITION_TO_PROCESS = 'closed_baseline_epoch'
    
    # Set to True to run on just one pair for testing
    testing = False
    
    # --- End Configuration ---
    
    if testing:
        all_pairs = [2010] # Process only the first pair for testing

    for pair_n in all_pairs:
        pipeline = ConceptLabeling_Pipeline(
            pair_n, 
            input_folder, 
            save_folder, 
            condition=CONDITION_TO_PROCESS, 
            testing=testing
        )
        pipeline.run()

    print("\n\n=== All pairs processed. ===")