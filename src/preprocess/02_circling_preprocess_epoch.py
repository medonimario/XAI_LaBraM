import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# Define the base directory for your data
base_data_folder = "/Volumes/T9/Circling_dataset/ICA"
save_folder = "/Volumes/T9/Circling_dataset/Epochs"

all_pairs = [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200,
             2210, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370, 2380, 2390,
             2410, 2420, 2430, 2440, 2450, 2460, 2470, 2480, 2490, 2500, 2510, 2520, 2530]

# all_pairs = [2010, 2020, 2130]

# Define new event IDs for clarity
CLOSED_EPOCH_ID = 205
OPEN_EPOCH_ID = 251

# Define epoching parameters
EPOCH_LEN_SEC = 4.0
STEP_SEC = 2.0  # (for 50% overlap of a 4s epoch)
BASELINE_LEN_SEC = 120.0

print(f"=== Starting Hyperscanning Baseline Epoching ===")
print(f"Found {len(all_pairs)} pairs to process.")

for pair_n in all_pairs:
    print(f"\n--- Processing Pair {pair_n} ---")
    
    pair_folder = os.path.join(base_data_folder, f"pair{str(pair_n)}")
    file_a = os.path.join(pair_folder, 'a_ica_raw.fif')
    file_b = os.path.join(pair_folder, 'b_ica_raw.fif')

    try:
        # 1. Load data
        raw_a = mne.io.read_raw_fif(file_a, verbose=False)
        raw_b = mne.io.read_raw_fif(file_b, verbose=False)
        
        # 2. Find original events (using subject A as reference)
        events_a = mne.find_events(raw_a, stim_channel='Status', verbose=False)
        events_b = mne.find_events(raw_b, stim_channel='Status', verbose=False)
        sfreq = raw_a.info['sfreq']
        
        # Your check:
        if not np.array_equal(events_a, events_b):
            print(f"  WARNING: Events do not match for pair {pair_n}! Proceeding with A's events.")
            # We will use events_a as the reference for both
        
        # 3. Find baseline start triggers
        original_event_ids = {
            "beginning_of_closed_baseline": 5,
            "beginning_of_open_baseline": 51,
            "end_of_baseline": 50,
        }
        
        closed_start_events = events_a[events_a[:, 2] == original_event_ids["beginning_of_closed_baseline"]]
        open_start_events = events_a[events_a[:, 2] == original_event_ids["beginning_of_open_baseline"]]
        
        # Robustness checks
        if len(closed_start_events) == 0:
            print(f"  ERROR: Missing 'closed_baseline' (5) trigger. Skipping pair.")
            continue
        if len(open_start_events) == 0:
            print(f"  ERROR: Missing 'open_baseline' (51) trigger. Skipping pair.")
            continue
            
        # Get the first sample of the first trigger (assuming only one)
        closed_start_sample = closed_start_events[0, 0]
        open_start_sample = open_start_events[0, 0]
        
        print(f"  Found closed baseline start (5) at sample {closed_start_sample}")
        print(f"  Found open baseline start (51) at sample {open_start_sample}")

        # 4. Create new events for overlapping epochs
        # We want epoch starts at 0, 2, 4, ... 116 (relative to the trigger)
        # np.arange(start, stop, step)
        # The 'stop' is (120 - 4 + 0.1) = 116.1 to ensure 116 is included
        epoch_starts_sec = np.arange(0, BASELINE_LEN_SEC - EPOCH_LEN_SEC + 0.1, STEP_SEC)
        
        # Convert these times to sample offsets
        epoch_starts_samples = (epoch_starts_sec * sfreq).astype(int)
        
        new_events_list = []
        
        # Create events for closed baseline
        for offset in epoch_starts_samples:
            new_sample = closed_start_sample + offset
            new_events_list.append([new_sample, 0, CLOSED_EPOCH_ID])
            
        # Create events for open baseline
        for offset in epoch_starts_samples:
            new_sample = open_start_sample + offset
            new_events_list.append([new_sample, 0, OPEN_EPOCH_ID])
            
        # Convert to numpy array and sort by time (sample number)
        new_events = np.array(new_events_list)
        new_events = new_events[new_events[:, 0].argsort()]
        
        # 5. Print summary
        n_closed = np.sum(new_events[:, 2] == CLOSED_EPOCH_ID)
        n_open = np.sum(new_events[:, 2] == OPEN_EPOCH_ID)
        print(f"  Created {n_closed} events for 'closed_baseline' (ID {CLOSED_EPOCH_ID})")
        print(f"  Created {n_open} events for 'open_baseline' (ID {OPEN_EPOCH_ID})")

        # 6. Plot the events for visual check
        plt.figure(figsize=(16, 5))
        
        # Plot original triggers
        plt.scatter(events_a[:, 0] / sfreq, events_a[:, 2], 
                    c='black', marker='x', s=150, label='Original Triggers (5, 50, 51)', zorder=10)
        
        # Plot new epoch events
        closed_mask = new_events[:, 2] == CLOSED_EPOCH_ID
        open_mask = new_events[:, 2] == OPEN_EPOCH_ID
        
        plt.scatter(new_events[closed_mask, 0] / sfreq, new_events[closed_mask, 2], 
                    c='blue', marker='o', s=40, alpha=0.7, label=f'Closed Epochs (ID {CLOSED_EPOCH_ID})')
        plt.scatter(new_events[open_mask, 0] / sfreq, new_events[open_mask, 2], 
                    c='red', marker='o', s=40, alpha=0.7, label=f'Open Epochs (ID {OPEN_EPOCH_ID})')
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Event ID')
        plt.title(f'Event Structure for Pair {pair_n}')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(-10, OPEN_EPOCH_ID + 50) # Adjust ylim to see all events
        plt.show()

        # 7. Create the MNE Epochs objects
        event_id_map = {
            'closed_baseline_epoch': CLOSED_EPOCH_ID,
            'open_baseline_epoch': OPEN_EPOCH_ID
        }
        # tmin=0.0 and tmax=4.0 because our new events mark the *start* of the 4s window
        tmin, tmax = 0.0, EPOCH_LEN_SEC 
        
        epochs_a = mne.Epochs(raw_a, new_events, event_id=event_id_map, tmin=tmin, tmax=tmax, 
                              baseline=None, preload=True, verbose=False)
                              
        epochs_b = mne.Epochs(raw_b, new_events, event_id=event_id_map, tmin=tmin, tmax=tmax, 
                              baseline=None, preload=True, verbose=False)
                              
        print(f"\n  Successfully created Epochs object for subject A:")
        print(epochs_a)
        print(f"  Successfully created Epochs object for subject B:")
        print(epochs_b)

        # 8. Downsample to 256 Hz
        epochs_a = epochs_a.resample(sfreq = 256)
        epochs_b = epochs_b.resample(sfreq = 256)

        # 9. Reference the data to average
        epochs_a.set_eeg_reference(ref_channels='average')
        epochs_b.set_eeg_reference(ref_channels='average')

        # 10. Save the epochs to disk
        save_path_a = os.path.join(save_folder, f"pair{pair_n}_a-epo.fif")
        save_path_b = os.path.join(save_folder, f"pair{pair_n}_b-epo.fif")

        epochs_a.save(save_path_a, overwrite=True)
        epochs_b.save(save_path_b, overwrite=True)
        
        print(f"--- Finished Pair {pair_n} ---")

    except FileNotFoundError:
        print(f"  ERROR: Data file not found for pair {pair_n}. Skipping.")
    except Exception as e:
        print(f"  An UNEXPECTED ERROR occurred for pair {pair_n}: {e}. Skipping.")

print("\n=== All pairs processed. ===")