import traceback
from isicpy.utils import load_synced_mat, closestSeries, makeFilterCoeffsSOS
from isicpy.lookup_tables import emg_montages, kinematics_offsets
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
from scipy import signal
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
        folder_name, list_of_blocks=[4]):
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=True)
    verbose = 0
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
    this_emg_montage = emg_montages['lower_v2']
    filterOpts = {
        'low': {
            'Wn': 250.,
            'N': 4,
            'btype': 'low',
            'ftype': 'butter'
        }
    }
    for idx_into_list, block_idx in enumerate(list_of_blocks):
        file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=False, stim_info_traces=False,
            load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True,
            load_vicon=True, vicon_as_df=True, interpolate_emg=True,
            )
        # TODO: fix error when loading lfp
        if data_dict['vicon'] is not None:
            emg_df = data_dict['vicon']['EMG'].copy()
            emg_df.rename(columns=this_emg_montage, inplace=True)
            emg_df.drop(columns=['NA'], inplace=True)
            emg_sample_rate = np.median(np.diff(emg_df.index.get_level_values('time_usec') * 1e-6)) ** -1
            filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
            emg_signals = signal.sosfiltfilt(filterCoeffs, (emg_df - emg_df.mean()).abs(), axis=0)
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
        if 'stim_info_traces' in data_dict:
            if data_dict['stim_info_traces'] is not None:
                stim_info_trace_df = pd.concat(data_dict['stim_info_traces'], names=['feature'], axis='columns')
                stim_info_trace_df.columns = stim_info_trace_df.columns.to_frame().apply(lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
                stim_info_trace_sample_rate = np.median(np.diff(stim_info_trace_df.index.get_level_values('time_usec') * 1e-6)) ** -1
                t_start = stim_info_trace_df.index.get_level_values('time_usec')[0] * 1e-6
                # pdb.set_trace()
                stim_info_trace_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    stim_info_trace_df.to_numpy(), stim_info_trace_sample_rate, t_start, channel_names=stim_info_trace_df.columns)
                stim_info_trace_signals_view = ephyviewer.TraceViewer(source=stim_info_trace_signals_source, name=f'block_{block_idx:0>2d}_stim_info_traces')
                if idx_into_list == 0:
                    win.add_view(stim_info_trace_signals_view, split_with=top_level_emg_view, orientation='vertical')
                    top_level_stim_info_trace_view = f'block_{block_idx:0>2d}_stim_info_traces'
                else:
                    win.add_view(stim_info_trace_signals_view, tabify_with=top_level_stim_info_trace_view)
        if data_dict['ripple'] is not None:
            if data_dict['ripple']['NEV'] is not None:
                nev_spikes_df = data_dict['ripple']['NEV']
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
        if (data_dict['vicon'] is not None):
            if ('Devices' in data_dict['vicon']) and False:
                devices_df = data_dict['vicon']['Devices']
                devices_df.columns.name = 'label'
                devices_signals = devices_df.to_numpy()
                devices_sample_rate = (np.median(
                    np.diff(devices_df.index.get_level_values('time_usec'))) * 1e-6) ** -1
                t_start = devices_df.index.get_level_values('time_usec')[0] * 1e-6
                devices_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    devices_signals, devices_sample_rate, t_start,
                    channel_names=[f'{cn}' for cn in devices_df.columns])
                devices_signals_view = ephyviewer.TraceViewer(
                    source=devices_signals_source, name=f'block_{block_idx:0>2d}_devices')
                win.add_view(devices_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
            if ('Points' in data_dict['vicon']):
                this_kin_offset = kinematics_offsets[folder_name][block_idx]
                points_df = data_dict['vicon']['Points']
                points_df.index += int(this_kin_offset)
                label_mask = points_df.columns.get_level_values('label').str.contains('ForeArm')
                for extra_label in ['Elbow', 'Foot', 'UpperArm', 'Hip', 'Knee', 'Ankle']:
                    label_mask = label_mask | points_df.columns.get_level_values('label').str.contains(extra_label)
                points_df = points_df.loc[:, label_mask].copy()
                points_df.interpolate(inplace=True)
                angles_dict = {
                    'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
                    'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
                    'LeftKnee': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
                    'RightKnee': ['RightHip', 'RightKnee', 'RightAnkle'],
                }
                for angle_name, angle_labels in angles_dict.items():
                    vec1 = (
                            points_df.xs(angle_labels[0], axis='columns', level='label') -
                            points_df.xs(angle_labels[1], axis='columns', level='label')
                    )
                    vec2 = (
                            points_df.xs(angle_labels[2], axis='columns', level='label') -
                            points_df.xs(angle_labels[1], axis='columns', level='label')
                    )
                    points_df.loc[:, (angle_name, 'angle')] = vg.angle(vec1.to_numpy(), vec2.to_numpy())
                lengths_dict = {
                    'LeftLimb': ['LeftHip', 'LeftFoot'],
                    'RightLimb': ['RightHip', 'RightFoot'],
                }
                for length_name, length_labels in lengths_dict.items():
                    vec1 = points_df.xs(length_labels[0], axis='columns', level='label')
                    vec2 = points_df.xs(length_labels[1], axis='columns', level='label')
                    points_df.loc[:, (length_name, 'length')] = vg.euclidean_distance(vec1.to_numpy(), vec2.to_numpy())

                points_df.columns = points_df.columns.to_frame().apply(
                    lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
                points_df.columns.name = 'label'
                points_df = points_df.fillna(method='ffill').fillna(method='bfill')
                points_df.drop(
                    columns=[
                        nm
                        for nm in points_df.columns
                        if ('_x' in nm) or ('_y' in nm) or ('_z' in nm)],
                    inplace=True
                )
                points_signals = points_df.to_numpy()
                points_sample_rate = (
                    (
                        np.median(
                            np.diff(points_df.index.get_level_values('time_usec'))) * 1e-6
                    ) ** -1
                )
                t_start = points_df.index.get_level_values('time_usec')[0] * 1e-6
                points_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    points_signals, points_sample_rate, t_start, channel_names=points_df.columns)
                points_signals_view = ephyviewer.TraceViewer(
                    source=points_signals_source, name=f'block_{block_idx:0>2d}_points')
                win.add_view(points_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
    win.show()
    app.exec_()
    return


if __name__ == '__main__':
    folder_name = "Day12_PM"
    list_of_blocks = [3]
    # folder_name = "Day8_AM"
    # list_of_blocks = [4]
    visualize_dataset(folder_name, list_of_blocks=list_of_blocks)
