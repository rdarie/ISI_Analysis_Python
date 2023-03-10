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
import vg
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
    font_scale=2, color_codes=True, rc=snsRCParams
)
for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV


def visualize_dataset(
        data_path, list_of_blocks=[4]):
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True)
    verbose = 0
    standardize_emg = False
    if standardize_emg:
        emg_scaler_path = data_path / "pickles" / "emg_scaler.p"
        with open(emg_scaler_path, 'rb') as handle:
            scaler = pickle.load(handle)
    this_emg_montage = emg_montages['lower']

    for idx_into_list, block_idx in enumerate(list_of_blocks):
        file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=False,
            load_ripple=True, ripple_variable_names=['NEV', 'NS5'], ripple_as_df=True,
            load_vicon=True, vicon_as_df=True,
            load_meta=True)
        # TODO: fix error when loading lfp
        if data_dict['vicon'] is not None:
            emg_df = data_dict['vicon']['EMG'].iloc[:, [ii for ii in range(12)] + [15]].copy()
            emg_df.columns = this_emg_montage + ['Sync']
            emg_df.columns.name = 'label'
            emg_signals = emg_df.to_numpy()
            emg_sample_rate = np.median(np.diff(emg_df.index.get_level_values('time_usec') * 1e-6)) ** -1
            t_start = emg_df.index.get_level_values('time_usec')[0] * 1e-6
            emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                emg_signals, emg_sample_rate, t_start, channel_names=emg_df.columns)
            emg_signals_view = ephyviewer.TraceViewer(source=emg_signals_source, name=f'block_{block_idx:0>2d}_emg')
            if idx_into_list == 0:
                win.add_view(emg_signals_view)
                top_level_emg_view = f'block_{block_idx:0>2d}_emg'
            else:
                win.add_view(emg_signals_view, tabify_with=top_level_emg_view)
        if data_dict['stim_info'] is not None:
            data_dict['stim_info'].loc[:, 'elecConfig_str'] = data_dict['stim_info'].apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
            ##
            stim_event_dict = {
                'label': data_dict['stim_info'].apply(
                    lambda x: f'\n{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}', axis='columns').to_numpy(),
                'time': (data_dict['stim_info'].index.get_level_values('timestamp_usec') * 1e-6).to_numpy(),
                'name': f'block_{block_idx:0>2d}_stim_info'
            }
            stim_event_original_dict = {
                'label': data_dict['stim_info'].apply(lambda x: f'\n{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}', axis='columns').to_numpy(),
                'time': (data_dict['stim_info']['original_timestamp_usec'] * 1e-6).to_numpy(),
                'name': f'block_{block_idx:0>2d}_original_stim_info'
            }
            event_source = ephyviewer.InMemoryEventSource(all_events=[stim_event_dict, stim_event_original_dict])
            event_view = ephyviewer.EventList(source=event_source, name=f'block_{block_idx:0>2d}_stim_info')
            if idx_into_list == 0:
                win.add_view(event_view, split_with=top_level_emg_view, orientation='horizontal')
                top_level_event_view = f'block_{block_idx:0>2d}_stim_info'
            else:
                win.add_view(event_view, tabify_with=top_level_event_view)
        if data_dict['ripple'] is not None:
            if data_dict['ripple']['NEV'] is not None:
                nev_spikes_df = data_dict['ripple']['NEV']  ## .loc[data_dict['ripple']['NEV']['Electrode'] >= 5120, :]
                spike_list = []
                for elec, elec_spikes in nev_spikes_df.groupby('Electrode'):
                    spike_list.append({
                        'name': f"{elec}",
                        'time': elec_spikes['time_seconds'].to_numpy()
                    })
                spike_source = ephyviewer.InMemorySpikeSource(all_spikes=spike_list)
                spike_view = ephyviewer.SpikeTrainViewer(source=spike_source, name=f'block_{block_idx:0>2d}_nev_spikes')
                if idx_into_list == 0:
                    win.add_view(spike_view, split_with=top_level_emg_view, orientation='vertical')
                    top_level_spike_view = f'block_{block_idx:0>2d}_nev_spikes'
                else:
                    win.add_view(spike_view, tabify_with=top_level_spike_view)
            if 'NF7' in data_dict['ripple']:
                if data_dict['ripple']['NF7'] is not None:
                    lfp_df = data_dict['ripple']['NF7'].iloc[:, :2].copy()

                    del data_dict['ripple']['NF7']
                    gc.collect()

                    lfp_signals = lfp_df.to_numpy()
                    lfp_sample_rate = 15e3
                    t_start = lfp_df.index.get_level_values('time_usec')[0] * 1e-6
                    lfp_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                        lfp_signals, lfp_sample_rate, t_start, channel_names=lfp_df.columns)
                    lfp_signals_view = ephyviewer.TraceViewer(source=lfp_signals_source, name=f'block_{block_idx:0>2d}_lfp')
                    if idx_into_list == 0:
                        win.add_view(lfp_signals_view, split_with=top_level_emg_view, orientation='vertical')
                        top_level_lfp_view = f'block_{block_idx:0>2d}_lfp'
                    else:
                        win.add_view(lfp_signals_view, tabify_with=top_level_lfp_view)
            if 'NS5' in data_dict['ripple']:
                if data_dict['ripple']['NS5'] is not None:
                    lfp_df = data_dict['ripple']['NS5'].iloc[:, :2].copy()

                    del data_dict['ripple']['NS5']
                    gc.collect()

                    lfp_signals = lfp_df.to_numpy()
                    lfp_sample_rate = 30e3
                    t_start = lfp_df.index.get_level_values('time_usec')[0] * 1e-6
                    lfp_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                        lfp_signals, lfp_sample_rate, t_start, channel_names=lfp_df.columns)
                    lfp_signals_view = ephyviewer.TraceViewer(
                        source=lfp_signals_source, name=f'block_{block_idx:0>2d}_ns5')
                    if idx_into_list == 0:
                        win.add_view(
                            lfp_signals_view, split_with=top_level_emg_view, orientation='vertical')
                        top_level_lfp_view = f'block_{block_idx:0>2d}_ns5'
                    else:
                        win.add_view(lfp_signals_view, tabify_with=top_level_lfp_view)
        if data_dict['vicon'] is not None:
            # pdb.set_trace()
            if 'Devices' in data_dict['vicon']:
                devices_df = data_dict['vicon']['Devices']
                devices_df.columns.name = 'label'
                devices_signals = devices_df.to_numpy()
                devices_sample_rate = (np.median(
                    np.diff(devices_df.index.get_level_values('time_usec'))) * 1e-6) ** -1
                t_start = devices_df.index.get_level_values('time_usec')[0] * 1e-6
                devices_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    devices_signals, devices_sample_rate, t_start,
                    channel_names=[f'{cn}' for cn in devices_df.columns])
                devices_signals_view = ephyviewer.TraceViewer(source=devices_signals_source,
                                                              name=f'block_{block_idx:0>2d}_devices')
                win.add_view(devices_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
            if 'Points' in data_dict['vicon']:
                points_df = data_dict['vicon']['Points']
                points_df.columns.get_level_values('label').unique()
                angles_dict = {
                    'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
                    'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
                }
                for angle_name, angle_labels in angles_dict.items():
                    vec1 = points_df.xs(angle_labels[0], axis='columns', level='label') - points_df.xs(
                        angle_labels[1], axis='columns', level='label')
                    vec2 = points_df.xs(angle_labels[2], axis='columns', level='label') - points_df.xs(
                        angle_labels[1], axis='columns', level='label')
                    points_df.loc[:, (angle_name, 'angle')] = vg.angle(vec1.to_numpy(), vec2.to_numpy())
                points_df.columns = data_dict['vicon']['Points'].columns.to_frame().apply(
                    lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
                points_df.columns.name = 'label'
                points_signals = points_df.to_numpy()
                points_sample_rate = (np.median(
                    np.diff(points_df.index.get_level_values('time_usec'))) * 1e-6) ** -1
                t_start = points_df.index.get_level_values('time_usec')[0] * 1e-6
                points_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    points_signals, points_sample_rate, t_start, channel_names=points_df.columns)
                points_signals_view = ephyviewer.TraceViewer(source=points_signals_source,
                                                             name=f'block_{block_idx:0>2d}_points')
                win.add_view(points_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
    win.show()
    app.exec_()
    return


if __name__ == '__main__':
    this_data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day12_PM")
    visualize_dataset(this_data_path)
