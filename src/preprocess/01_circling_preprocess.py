import math
#import matplotlib
import matplotlib.pyplot as plt
import pathlib
import mne
import numpy as np
import pandas as pd
import glob, os
import pickle
from mne.preprocessing import (ICA, corrmap,find_bad_channels_lof)
from autoreject import (AutoReject,get_rejection_threshold)


pairs = [2010,2020,2030,2040,2050,2060,2070,2080,2090,2100,2110,2120,2130,2140,2150,2160,2170,2180,2190,2200,
            2210,2220,2230,2240,2250,2260,2270,2280,2290,2300,2310,2320,2330,2340,2350,2360,2370,2380,2390,
            2410,2420,2430,2440,2450,2460,2470,2480,2490,2500,2510,2520,2530]

for pair_n in pairs:
    folder = f"/Users/aldab/Documents/circling_experiment_data/pair{str(pair_n)}"
    os.chdir(folder)

    raw = mne.io.read_raw_bdf(glob.glob("*.bdf")[0], verbose=False)

    ### retrieve channel names
    channel_names = raw.ch_names

    ### pick channels
    channels_a = mne.pick_channels_regexp(channel_names, '(1\-.*)|(1\-.*)|(Status)')
    channels_b = mne.pick_channels_regexp(channel_names, '(2\-.*)|(2\-.*)|(Status)')

    ### separate the data
    raw_a = raw.copy().pick(channels_a).set_channel_types({'1-EXG1':'eog','1-EXG2':'eog','1-EXG3':'eog','1-EXG4':'ecg',
                                                        '1-EXG5':'ecg','1-EXG6':'emg','1-EXG7':'emg','1-EXG8':'emg'})
    raw_b = raw.copy().pick(channels_b).set_channel_types({'2-EXG1':'eog','2-EXG2':'eog','2-EXG3':'eog','2-EXG4':'ecg',
                                                        '2-EXG5':'ecg','2-EXG6':'emg','2-EXG7':'emg','2-EXG8':'emg'})

    ### rename channels  
    biosemi_layout = mne.channels.make_standard_montage('biosemi64')
    biosemi_names = biosemi_layout.ch_names

    for i in range(len(biosemi_names)):
        raw_a.rename_channels(mapping = {raw_a.ch_names[i]:biosemi_names[i]})

    for i in range(len(biosemi_names)):
        raw_b.rename_channels(mapping = {raw_b.ch_names[i]:biosemi_names[i]})