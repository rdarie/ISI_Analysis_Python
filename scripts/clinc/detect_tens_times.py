import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_utils import parse_mb_stim_csv, assign_tens_metadata
from isicpy.clinc_lookup_tables import dsi_trig_sample_rate, sid_to_cactus, cactus_to_sid, dsi_mb_clock_offsets
import numpy as np
from scipy import signal
import yaml
import os

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]

for file_name in file_name_list:
    print(f"Processing {file_name}...")
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))

    with open(folder_path / 'dsi_block_lookup.json', 'r') as f:
        emg_block_name = json.load(f)[file_name][0]

    dsi_trigs = pd.read_parquet(folder_path / (emg_block_name + '_dsi_trigs.parquet'))
    clock_difference = dsi_mb_clock_offsets[folder_path.stem]
    with open(folder_path / 'dsi_to_mb_fine_offsets.json', 'r') as f:
        dsi_fine_offset = json.load(f)[file_name][emg_block_name]
    dsi_total_offset = pd.Timedelta(clock_difference + dsi_fine_offset, unit='s')
    print(f'DSI offset = {clock_difference} + {dsi_fine_offset:.3f} = {dsi_total_offset.total_seconds():.3f}')
    dsi_trigs.index = dsi_trigs.index + dsi_total_offset

    t_zero = clinc_df.index[0]
    dsi_trigs.index -= t_zero
    clinc_df.index -= t_zero

    signal_thresh = 2.5
    temp = pd.Series(dsi_trigs['PhoenixLeft867-2:Input'].to_numpy())

    cross_index, cross_mask = getThresholdCrossings(
        temp, thresh=signal_thresh, fs=dsi_trig_sample_rate, iti=0.1, absVal=False, keep_max=False)
    align_timestamps = dsi_trigs.index[cross_mask].copy()

    with open(folder_path / 'tens_info.json', 'r') as f:
        tens_info_list = json.load(f)[file_name]

    meta_list = []
    for ts in align_timestamps:
        ts_sec = ts.total_seconds()
        these_params = assign_tens_metadata(ts_sec, tens_info_list).to_frame(name=ts + t_zero).T
        meta_list.append(these_params)

    tens_info = pd.concat(meta_list, names=['timestamp'])
    tens_info.index.name = 'timestamp'
    output_filename = file_name + '_tens_info.parquet'
    tens_info.to_parquet(folder_path / output_filename, engine='fastparquet')
