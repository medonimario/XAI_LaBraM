# --- START OF ADAPTED 01_preprocess_ica.py ---

import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mne.preprocessing import ICA
import mne_icalabel

# We assume the 'logger' module exists in the same directory
# as you are running this script.
from logger import Logger

class ICA_Pipeline():
    def __init__(self, pair_n, data_path, save_path, testing=False):
        self.pair_n = str(pair_n)
        self.data_path = data_path  # This is now the base folder, e.g., /.../circling_eeg_data
        self.save_path = save_path  # Base save path
        self.testing = testing
        
        self.raw_a = None
        self.raw_b = None
        self.sfreq = None
        self.bads_a = []
        self.bads_b = []
        
        # Initialize the logger for the pair
        log_base_path = os.path.join(self.save_path, 'logs')
        self.logger = Logger(f"pair_{self.pair_n}", log_base_path)

    def load_and_crop_data(self):
        """
        Loads the pre-filtered a_filtered_raw.fif and b_filtered_raw.fif
        files and crops them 5 seconds after the final '50' event.
        """
        self.logger.log_section(f"Load and Crop Data for Pair {self.pair_n}")
        
        try:
            folder = os.path.join(self.data_path, f"pair{self.pair_n}")
            file_a = os.path.join(folder, 'a_filtered_raw.fif')
            file_b = os.path.join(folder, 'b_filtered_raw.fif')
            
            self.raw_a = mne.io.read_raw_fif(file_a, preload=True)
            self.raw_b = mne.io.read_raw_fif(file_b, preload=True)
            
            self.sfreq = self.raw_a.info['sfreq']
            self.logger.log_text(f"Loaded raw data for subject A from {file_a}")
            self.logger.log_text(f"Loaded raw data for subject B from {file_b}")
            self.logger.log_text(f"Sampling frequency: {self.sfreq} Hz")

            # Find events in subject A (assuming they are the same for B)
            events = mne.find_events(self.raw_a, stim_channel='Status', verbose=False)
            
            # Find the last '50' event
            end_baseline_events = events[events[:, 2] == 50]
            if len(end_baseline_events) == 0:
                self.logger.log_text("WARNING: No '50' (end_of_baseline) event found. Using full file length.")
                crop_time_sec = self.raw_a.times[-1]
            else:
                last_50_event_sample = end_baseline_events[-1, 0]
                crop_time_sec = (last_50_event_sample / self.sfreq) + 5.0 # 5s buffer
                self.logger.log_text(f"Found last '50' event at {last_50_event_sample / self.sfreq:.2f}s.")
                self.logger.log_text(f"Cropping both files at {crop_time_sec:.2f}s (event + 5s buffer).")

            # Crop both raw objects to this time
            self.raw_a.crop(tmax=crop_time_sec)
            self.raw_b.crop(tmax=crop_time_sec)

            self.logger.log_text(f"Data length after cropping: {self.raw_a.times[-1]/60:.2f} minutes")
            print(f"Data length after cropping: {self.raw_a.times[-1]/60:.2f} minutes")

        except FileNotFoundError as e:
            self.logger.log_text(f"ERROR: Data file not found. {e}")
            raise

    def remove_bad_channels(self, raw, subject_label):
        """
        Shows the interactive plot to mark bad channels.
        """
        self.logger.log_section(f"Remove Bad Channels - Subject {subject_label.upper()}")
        print(f"--- Please inspect and mark bad channels for Subject {subject_label.upper()} ---")
        
        mne.viz.plot_raw(raw, duration=30, n_channels=64, scalings=None, block=True, theme="light")
        
        # After the plot is closed, the bad channels are marked in raw.info['bads']
        bad_channels = raw.info['bads']
        self.logger.log_text(f"Marked bad channels: {bad_channels}")
        print(f"Marked bad channels for Subject {subject_label.upper()}: {bad_channels}")
        
        # Create a copy excluding bads for ICA fitting
        good_eeg = raw.copy().pick_types(eeg=True, exclude='bads')
        
        return good_eeg, bad_channels
    
    def run_ica(self, raw, good_eeg, subject_label):
        """
        Runs the full ICA and labeling pipeline on a given raw object.
        Applies the ICA to the 'raw' object in place.
        """
        self.logger.log_section(f"Run ICA - Subject {subject_label.upper()}")
        
        # 1. Run ICA
        random_state = 42
        # Ensure n_components is not more than available channels in good_eeg
        n_channels = len(good_eeg.info['ch_names'])
        n_components = min(n_channels, 64) # 64 was in your original script
        
        # MNE recommendation: n_components = n_channels - n_bad_channels
        # We can also use a float (e.g., 0.99) or 'mle'
        # Let's use the explicit channel count from the 'good_eeg' copy
        n_components = n_channels 
        
        print(f"Number of components for subject {subject_label.upper()}: {n_components}")
        self.logger.log_text(f"Number of components: {n_components}")
        
        ica = ICA(n_components=n_components, method="fastica", max_iter="auto", random_state=random_state)

        # Fit ICA on the 'good_eeg' copy (bads removed)
        # We fit on the whole (cropped) file, decimating for speed
        ica.fit(good_eeg, decim=4)
        
        self.logger.log_text(f"ICA parameters: n_components={n_components}, method='fastica', max_iter='auto', random_state={random_state}, decim=4")
        print(f"ICA fitted for subject {subject_label.upper()}.")

        # 2. Automatic Labeling
        ic_labels = mne_icalabel.label_components(good_eeg, ica, method="iclabel")
        self.logger.log_text("Automatic ICA labeling completed using iclabel method.")

        # # <<< CRITICAL FIX: Automatically set ica.exclude based on labels
        # ica.exclude = [i for i, label in enumerate(ic_labels['labels']) if label != 'brain']
        # self.logger.log_text(f"Auto-excluding non-brain components: **{ica.exclude}**")
        # print(f"Auto-excluding components for subject {subject_label.upper()}: {ica.exclude}")

        # 3. Get ICA component predicted probabilities
        y_pred_proba, labels = self.compute_ica_component_probabilities(ic_labels)
        probability_plot = self.plot_ica_component_probabilities(y_pred_proba, labels)
        probability_plot.show(block=False)
        # Save the probability plot
        plot_name = f"ica_component_probabilities_{subject_label}"
        self.logger.save_plot(probability_plot.gcf(), plot_name)

        # 4. Plot ICA sources (optional, can be noisy)
        ica.plot_sources(good_eeg, picks=range(30), show=True, block=True, theme="light")

        plt.close("all")

        # 5. Save the plots for the excluded components
        self.logger.log_text(f"ICA components excluded: **{ica.exclude}**")
        self.save_excluded_plots(ica, ic_labels, good_eeg, subject_label)

        # 6. Apply ICA to the *original* raw data
        print(f"Applying ICA to raw data for subject {subject_label.upper()}...")
        ica.apply(raw)
        
        print(f"--- Please review ICA-cleaned data for Subject {subject_label.upper()} ---")
        mne.viz.plot_raw(raw, duration=30, n_channels=64, scalings=None, block=True, theme="light")

    # --- Helper Methods (Copied from your original script) ---
    # (No changes needed, except for 'save_excluded_plots')

    def compute_ica_component_probabilities(self, ica_dict):
        """
        Extracts the probability predictions and labels from the provided dictionary.

        Parameters:
        - ica_dict: dict, dictionary containing 'y_pred_proba' and 'labels'.

        Returns:
        - y_pred_proba: list, predicted probabilities of each IC component.
        - labels: list, labels corresponding to each IC component.
        """
        y_pred_proba = ica_dict['y_pred_proba']
        labels = ica_dict['labels']
        return y_pred_proba, labels

    def plot_ica_component_probabilities(self, y_pred_proba, labels, n=30):
        """
        Plots the probabilities of each ICA component label.

        Parameters:
        - y_pred_proba: list, predicted probabilities of each IC component.
        - labels: list, labels corresponding to each IC component.
        """
        y_pred_proba = y_pred_proba[:n]
        labels = labels[:n]

        # Generating IC labels
        ic_labels = [f'IC{i}' for i in range(len(labels))]

        # Plotting the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ic_labels, y_pred_proba, color='lightblue', alpha=0.8)

        # Coloring the bars according to their labels
        colors = {'brain': 'green', 'eye blink': 'orange', 'muscle artifact': 'red', 'other': 'gray'}
        for bar, label in zip(bars, labels):
            bar.set_color(colors.get(label, 'lightblue'))

        # Adding labels and title
        plt.xlabel('Independent Component (IC)')
        plt.ylabel('Probability')
        plt.title('Probability of Each ICA Component Label')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y')

        # Adding a legend
        legend_elements = [Line2D([0], [0], color='green', lw=4, label='Brain'),
                        Line2D([0], [0], color='orange', lw=4, label='Eye Blink'),
                        Line2D([0], [0], color='red', lw=4, label='Muscle Artifact'),
                        Line2D([0], [0], color='gray', lw=4, label='Other')]
        plt.legend(handles=legend_elements, title="Labels")
        plt.tight_layout()
        return plt

    def save_excluded_plots(self, ica, ic_labels, good_eeg, subject_label):
            excluded_indices = ica.exclude
            print(f"Saving excluded component plots for subject {subject_label.upper()}...")
            self.logger.log_subsection(f"Excluded components - Subject {subject_label.upper()}")
            
            if not excluded_indices:
                self.logger.log_text("No components were excluded.")
                print("No components were excluded.")
                return

            try:
                excluded_plot = ica.plot_sources(good_eeg, picks=excluded_indices, show=False, theme="light")
                plot_name = f"excluded_components_{subject_label}"
                self.logger.save_plot(excluded_plot, plot_name)
                plt.close(excluded_plot) # Close the figure
            except Exception as e:
                self.logger.log_text(f"Could not plot excluded sources: {e}")

            for idx in excluded_indices:
                self.logger.log_text(f"IC **{idx}** labeled as **'{ic_labels['labels'][idx]}'**, with a probability of **{ic_labels['y_pred_proba'][idx]:.2f}**")
                
                try:
                    # Plot the component for inspection
                    fig = ica.plot_properties(good_eeg, picks=idx, show=False, verbose=False)
                    # Save the plot
                    plot_name = f"ica_component_{idx}_properties_{subject_label}"
                    self.logger.save_plot(fig, plot_name)
                    # close the plot
                    plt.close(fig)
                except Exception as e:
                     self.logger.log_text(f"Could not plot properties for IC {idx}: {e}")
            print("Saved plots.")
    
    def interpolate_bad_channels(self, raw, bad_channels, subject_label):
        self.logger.log_section(f"Interpolate Bad Channels - Subject {subject_label.upper()}")
        # Restore the bads list to the raw object
        raw.info['bads'] = bad_channels
        if bad_channels:
            raw.interpolate_bads(reset_bads=True)
            self.logger.log_text(f"Interpolated bad channels: {bad_channels}")
        else:
            self.logger.log_text("No bad channels to interpolate.")

    def save_ica_raw(self, raw, subject_label):
        # Save inside a dedicated folder for this pair
        save_dir = os.path.join(self.save_path, f"pair{self.pair_n}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as a_{pair_n}_ica-raw.fif
        save_path_raw = os.path.join(save_dir, f"{subject_label}_ica_raw.fif")
        
        self.logger.log_section(f"Save ICA-Cleaned Raw Data - Subject {subject_label.upper()}")
        raw.save(save_path_raw, overwrite=True)
        self.logger.log_text(f"Saved ICA-cleaned raw data to {save_path_raw}")

    
    def run(self):
        print("\n\n")
        print("#"*50)
        print(f"Processing Pair {self.pair_n}")
        print("Step1: Loading and Cropping data...")
        print("#"*50)
        self.load_and_crop_data()

        if self.testing:
            # crop data to 5 minutes for testing
            self.raw_a.crop(tmax=300)
            self.raw_b.crop(tmax=300)
            self.logger.log_text("Data cropped to 5 minutes for testing.")

        # --- Process Subject A ---
        print("\n\n")
        print("#"*50)
        print("Step 2a: Removing bad channels (Subject A)...")
        print("#"*50)
        good_a, self.bads_a = self.remove_bad_channels(self.raw_a, 'a')

        print("\n\n")
        print("#"*50)
        print("Step 3a: Running ICA (Subject A)...")
        print("#"*50)
        self.run_ica(self.raw_a, good_a, 'a')

        # --- Process Subject B ---
        print("\n\n")
        print("#"*50)
        print("Step 2b: Removing bad channels (Subject B)...")
        print("#"*50)
        good_b, self.bads_b = self.remove_bad_channels(self.raw_b, 'b')

        print("\n\n")
        print("#"*50)
        print("Step 3b: Running ICA (Subject B)...")
        print("#"*50)
        self.run_ica(self.raw_b, good_b, 'b')

        # --- Interpolate and Save ---
        
        print("\n\n")
        print("#"*50)
        print("Step 4a: Interpolating bad channels (Subject A)...")
        print("#"*50)
        self.interpolate_bad_channels(self.raw_a, self.bads_a, 'a')

        print("\n\n")
        print("#"*50)
        print("Step 4b: Interpolating bad channels (Subject B)...")
        print("#"*50)
        self.interpolate_bad_channels(self.raw_b, self.bads_b, 'b')

        print("\n\n")
        print("#"*50)
        print("Step 5a: Saving raw data (Subject A)...")
        print("#"*50)
        self.save_ica_raw(self.raw_a, 'a')
        
        print("\n\n")
        print("#"*50)
        print("Step 5b: Saving raw data (Subject B)...")
        print("#"*50)
        self.save_ica_raw(self.raw_b, 'b')
        
        # Convert final log
        self.logger.convert_markdown_to_html()


if __name__ == "__main__":
    # Your list of pairs from the previous script
    # all_pairs = [2020,2030,2040,2050,2060,2070,2080,2090,2100,2110,2120,2130,2140,2150,2160,2170,2180,2190,2200,
    #             2210,2220,2230,2240,2250,2260,2270,2280,2290,2300,2310,2320,2330,2340,2350,2360,2370,2380,2390,
    #             2410,2420,2430,2440,2450,2460,2470,2480,2490,2500,2510,2520,2530]
    
    all_pairs = [2370,2380,2390,
                2410,2420,2430,2440,2450,2460,2470,2480,2490,2500,2510,2520,2530]
    
    base_data_path = "/Users/s204684/mount/circling_eeg_data"
    # I'm creating a new 'preprocessed' subfolder for the outputs
    save_path = "/Volumes/T9/Circling_dataset/ICA" 
    
    # You can set this to True to test with just one pair
    testing = True
    
    if testing:
        all_pairs = [2180] # Process only the first pair for testing

    for pair_n in all_pairs:
        pipeline = ICA_Pipeline(pair_n, base_data_path, save_path, testing=False) # 'testing' flag in init is a bit redundant now, but kept
        pipeline.run()