import traceback
from isicpy.utils import load_synced_mat, closestSeries
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import os
import gc
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import ephyviewer
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
useDPI = 200
dpiFactor = 72 / useDPI

import seaborn as sns
from matplotlib import pyplot as plt

snsRCParams = {
        'figure.dpi': useDPI, 'savefig.dpi': useDPI,
        'lines.linewidth': .5,
        'lines.markersize': 2.5,
        'patch.linewidth': .5,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "axes.linewidth": .125,
        "grid.linewidth": .2,
        "font.size": 4,
        "axes.labelsize": 7,
        "axes.titlesize": 9,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 9,
        "xtick.bottom": True,
        "xtick.top": False,
        "ytick.left": True,
        "ytick.right": False,
        "xtick.major.width": .125,
        "ytick.major.width": .125,
        "xtick.minor.width": .125,
        "ytick.minor.width": .125,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
    }
mplRCParams = {
    'figure.titlesize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }
sns.set(
    context='talk', style='white',
    palette='dark', font='sans-serif',
    font_scale=2, color_codes=True, rc=snsRCParams)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV


def visualize_dataset(data_path):
    app = ephyviewer.mkQApp()

    verbose = 0
    standardize_emg = False
    if standardize_emg:
        emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
        with open(emg_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)

    this_emg_montage = emg_montages['lower']
    list_of_blocks = [2, 3]
    for idx_into_list, block_idx in enumerate(list_of_blocks):
        if idx_into_list == 0:
            # reference_time = pd.to_datetime(data_dict['meta']['Recording_Date'])
            time_offset = 0
        else:
            # time_offset = (pd.to_datetime(data_dict['meta']['Recording_Date']) - reference_time).total_seconds()
            time_offset = all_emg[list_of_blocks[idx_into_list - 1]].index.get_level_values('time_usec')[-1] + int(emg_sr ** -1 * 1e6)
            print(f'time_offset = {time_offset}')

        file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True,
            load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True,
            load_vicon=True, vicon_as_df=True,
            load_meta=True,
        )

        if data_dict['vicon'] is not None:
            all_emg[block_idx] = data_dict['vicon']['EMG'].iloc[:, :12].copy()
            all_emg[block_idx].columns = this_emg_montage
            all_emg[block_idx].columns.name = 'label'
            all_emg[block_idx].index = all_emg[block_idx].index + int(time_offset * 1e6)
            # pdb.set_trace()
            emg_sr = np.median(np.diff(all_emg[block_idx].index.get_level_values('time_usec')))

        if data_dict['stim_info'] is not None:
            all_stim_info[block_idx] = data_dict['stim_info']
            all_stim_info[block_idx].index = all_stim_info[block_idx].index + int(time_offset * 1e6)
            all_stim_info[block_idx].loc[:, 'time'] = all_stim_info[block_idx].loc[:, 'time'] + (time_offset)
            all_stim_info[block_idx].loc[:, 'original_timestamp_usec'] = all_stim_info[block_idx].loc[:, 'original_timestamp_usec'] + int(time_offset * 1e6)

            stim_event_dict = {
                'label': data_dict['stim_info'].apply(lambda x: f'{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}',
                                            axis='columns').to_numpy(),
                'time': (data_dict['stim_info'].index.get_level_values('timestamp_usec') * 1e-6).to_numpy(),
                'name': f'block_{0>2d:block_idx}_stim_info'
            }
            event_source = ephyviewer.InMemoryEventSource(all_events=[stim_event_dict])
            event_view = ephyviewer.EventList(source=event_source, name='stim_info')

        if data_dict['ripple'] is not None:
            if data_dict['ripple']['NEV'] is not None:
                all_nev_spikes[block_idx] = data_dict['ripple']['NEV'].loc[data_dict['ripple']['NEV']['Electrode'] >= 5120, :]
                all_nev_spikes[block_idx].loc[:, 'Electrode'] = all_nev_spikes[block_idx]['Electrode'] - 5120
                all_nev_spikes[block_idx].loc[:, 'time_usec'] = all_nev_spikes[block_idx].loc[:, 'time_usec'] + int(time_offset * 1e6)
                all_nev_spikes[block_idx].loc[:, 'time_seconds'] = all_nev_spikes[block_idx].loc[:, 'time_seconds'] + time_offset
            '''if data_dict['ripple']['NF7'] is not None:
                all_lfp[block_idx] = data_dict['ripple']['NF7']'''

    all_emg_df = pd.concat(all_emg)
    # all_lfp_df = pd.concat(all_lfp)
    stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
    stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}',axis='columns')
    nev_spikes_df = pd.concat(all_nev_spikes, names=['block', 'index'])

    stim_event_original = {
        'label': stim_info_df.apply(lambda x: f'{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}', axis='columns').to_numpy(),
        'time': (stim_info_df['original_timestamp_usec'] * 1e-6).to_numpy(),
        'name': 'stim_info_original'
        }

    event_source_o = ephyviewer.InMemoryEventSource(all_events=[stim_event_original])
    event_view_o = ephyviewer.EventList(source=event_source_o, name='stim_info_original')

    spike_list = []
    for elec, elec_spikes in nev_spikes_df.groupby('Electrode'):
        spike_list.append({
            'name': f"{elec}",
            'time': elec_spikes['time_seconds'].to_numpy()
        })
    spike_source = ephyviewer.InMemorySpikeSource(all_spikes=spike_list)
    spike_view = ephyviewer.SpikeTrainViewer(source=spike_source, name='nev_spikes')

    emg_signals = all_emg_df.to_numpy()
    emg_sample_rate = (np.median(np.diff(all_emg_df.index.get_level_values('time_usec'))) * 1e-6) ** (-1)
    t_start = all_emg_df.index.get_level_values('time_usec')[0] * 1e-6

    emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
        emg_signals, emg_sample_rate, t_start, channel_names=all_emg_df.columns)
    emg_signals_view = ephyviewer.TraceViewer(source=emg_signals_source, name='EMG')

    '''
    lfp_signals = all_lfp_df.to_numpy()
    lfp_sample_rate = 15e3
    t_start = all_lfp_df.index.get_level_values('time_usec')[0] * 1e-6

    lfp_signals_source = ephyviewer.InMemoryAnalogSignalSource(
        lfp_signals, lfp_sample_rate, t_start, channel_names=all_lfp_df.columns)
    lfp_signals_view = ephyviewer.TraceViewer(source=lfp_signals_source, name='lfp')
    '''

    win = ephyviewer.MainViewer(debug=True)
    win.add_view(emg_signals_view)
    win.add_view(event_view, split_with='EMG', orientation='horizontal')
    win.add_view(event_view_o, tabify_with='stim_info')
    # win.add_view(lfp_signals_view, split_with='EMG', orientation='vertical')
    win.add_view(spike_view, split_with='EMG', orientation='vertical')
    win.show()

    app.exec_()

if __name__=='__main__':
    data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day11_PM")
    visualize_dataset(data_path)
