import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoreject import AutoReject
from logger import Logger  # Assuming you have a logger.py file with the Logger class


class Rejecting_Pipeline():
    def __init__(self, pair_n, epochs_path, testing=False):
        self.pair_n = str(pair_n)
        self.epochs_path = epochs_path
        self.testing = testing

        # Define input/output paths
        self.epochs_a_path = os.path.join(self.epochs_path, f"pair{self.pair_n}_a-epo.fif")
        self.epochs_b_path = os.path.join(self.epochs_path, f"pair{self.pair_n}_b-epo.fif")

        # Initialize attributes
        self.epochs_a = None
        self.epochs_b = None
        
        # Initialize the logger
        log_base_path = os.path.join(self.epochs_path, 'logs')
        # Use a unique log file name for this step
        self.logger = Logger(f"pair_{self.pair_n}_03_rejection", log_base_path)

    def load_data(self):
        """Loads the epoched data from the previous script."""
        self.logger.log_section("Load Epoch Data for Rejection")

        try:
            # Load subject A epochs
            if not os.path.exists(self.epochs_a_path):
                raise FileNotFoundError(f"Epoch file not found at {self.epochs_a_path}. Please run script 02 first.")
            self.epochs_a = mne.read_epochs(self.epochs_a_path, preload=True)
            self.logger.log_text(f"Loaded epochs for subject A from {self.epochs_a_path}")
            self.logger.log_text(f"Epochs A info: {len(self.epochs_a)} epochs")

            # Load subject B epochs
            if not os.path.exists(self.epochs_b_path):
                raise FileNotFoundError(f"Epoch file not found at {self.epochs_b_path}. Please run script 02 first.")
            self.epochs_b = mne.read_epochs(self.epochs_b_path, preload=True)
            self.logger.log_text(f"Loaded epochs for subject B from {self.epochs_b_path}")
            self.logger.log_text(f"Epochs B info: {len(self.epochs_b)} epochs")

        except Exception as e:
            self.logger.log_text(f"ERROR: Failed to load data. {e}")
            raise

    def run_autoreject_on_epochs(self, epochs, subject_label):
        """
        Fits and applies AutoReject to an Epochs object and updates its metadata.
        """
        self.logger.log_subsection(f"Running Autoreject for Subject {subject_label.upper()}")
        
        # 1. Initialize AutoReject
        # n_interpolate=[0] means we only reject, not interpolate
        ar = AutoReject(n_interpolate=[0], picks="eeg", random_state=42)
        
        # 2. Fit AutoReject on the epochs
        # Since you have few epochs, we fit on all of them.
        ar.fit(epochs)
        self.logger.log_text(f"Fitted AutoReject.")
        
        # 3. Transform the epochs to get the rejection log
        # We save 'epochs_clean' just to get the log, but we will save the
        # original 'epochs' object with updated metadata.
        _, reject_log = ar.transform(epochs, return_log=True)
        self.logger.log_text(f"Transformed epochs. Plotting reject log.")

        # 4. Plot and log the rejection log
        plot_reject_log = reject_log.plot(orientation="horizontal", show=False)
        plot_name = f"autoreject_log_{subject_label}"
        self.logger.save_plot(plot_reject_log, plot_name)
        
        # 5. Get the boolean array of bad epochs
        bad_epochs_bool = reject_log.bad_epochs
        num_bad = np.sum(bad_epochs_bool)
        num_total = len(epochs)
        num_good = num_total - num_bad
        self.logger.log_text(f"Identified {num_bad}/{num_total} bad epochs.")
        self.logger.log_text(f"Remaining good epochs: {num_good}/{num_total}.")

        # 6. Add rejection info to metadata
        # Create metadata DataFrame if it doesn't exist
        if epochs.metadata is None:
            self.logger.log_text("No existing metadata found, creating new DataFrame.")
            epochs.metadata = pd.DataFrame(index=range(num_total))
            
        # Add the new 'autoreject_bad' column
        epochs.metadata['autoreject_bad'] = bad_epochs_bool
        self.logger.log_text(f"Added 'autoreject_bad' column to metadata.")

        # 7. Plot and save bad epochs
        if num_bad > 0:
            self.logger.log_text("Plotting and saving bad epochs for review...")
            try:
                # Plot all bad epochs
                bad_epochs_plot = epochs[bad_epochs_bool].plot(n_channels=64, scalings=None, show=False)
                self.logger.save_plot(bad_epochs_plot, f"autoreject_bad_epochs_plot_{subject_label}")
            except Exception as e:
                self.logger.log_text(f"Could not plot bad epochs: {e}")
        else:
            self.logger.log_text("No bad epochs identified.")
            
        return epochs


    def save_updated_epochs(self):
        """Saves the epoch files with updated metadata, overwriting the old files."""
        self.logger.log_section("Save Updated Epochs")
        
        try:
            # Save subject A
            self.epochs_a.save(self.epochs_a_path, overwrite=True)
            self.logger.log_text(f"Saved updated epochs with rejection info to {self.epochs_a_path}")
            
            # Save subject B
            self.epochs_b.save(self.epochs_b_path, overwrite=True)
            self.logger.log_text(f"Saved updated epochs with rejection info to {self.epochs_b_path}")
        
        except Exception as e:
            self.logger.log_text(f"ERROR: Failed to save epochs. {e}")
            raise

    def run(self):
        print("\n\n")
        print("#"*50)
        print(f"SCRIPT 3: REJECTING BAD EPOCHS FOR PAIR {self.pair_n}")
        print("#"*50)

        try:
            print("\nStep 1: Loading epoch data...")
            self.load_data()

            print("\nStep 2: Running AutoReject...")
            # Run for Subject A
            self.epochs_a = self.run_autoreject_on_epochs(self.epochs_a, 'a')
            # Run for Subject B
            self.epochs_b = self.run_autoreject_on_epochs(self.epochs_b, 'b')

            print("\nStep 3: Saving epochs with updated metadata...")
            self.save_updated_epochs()

            print(f"\nEpoch rejection script completed successfully for Pair {self.pair_n}!")
            self.logger.log_section("Preprocessing completed successfully!")

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
    
    # This is the 'save_folder' from your 02 script
    epochs_path = "/Volumes/T9/Circling_dataset/Epochs"
    
    # Set to True to run on just one pair for testing
    testing = False
    
    if testing:
        all_pairs = [2010] # Process only the first pair for testing

    for pair_n in all_pairs:
        pipeline = Rejecting_Pipeline(pair_n, epochs_path, testing=testing)
        pipeline.run()