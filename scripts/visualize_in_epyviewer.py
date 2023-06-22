import traceback
from isicpy.utils import load_synced_mat, closestSeries, makeFilterCoeffsSOS, timestring_to_timestamp
from isicpy.lookup_tables import emg_montages, kinematics_offsets, video_info
from pathlib import Path
import pandas as pd
import numpy as np
import cloudpickle as pickle
import pdb
import os
import gc
import vg
import av
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

from numpy.polynomial import Polynomial
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

def adjust_channel_name(cn):
    signal_type, ch_num_str = cn.split(' ')
    elec = int(ch_num_str)
    if elec < 128:
        return f"ch {elec} (caudal)"
    else:
        return f"ch {elec - 128} (rostral)"

def visualize_dataset(
        folder_name, list_of_blocks=[4]):
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=False)
    verbose = 0
    data_path = Path(f"/users/rdarie/scratch/3_Preprocessed_Data/{folder_name}")
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
        try:
            this_kin_offset = kinematics_offsets[folder_name][block_idx]
        except:
            this_kin_offset = 0
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=True, stim_info_traces=False, force_trains=True,
            load_ripple=True, ripple_variable_names=['NEV', 'NF7', 'TimeCode'], ripple_as_df=True,  # 'NS5', 'TimeCode'
            load_vicon=True, vicon_as_df=True, vicon_variable_names=['EMG'], interpolate_emg=True, kinematics_time_offset=this_kin_offset,  # , 'Points'
            load_all_logs=False, verbose=1
            )
        # TODO: fix error when loading lfp
        try:
            first_ripple = data_dict['ripple']['TimeCode'].iloc[0, :]
            print(f"First ripple timecode is {first_ripple['TimeString']}")
            video_times = []
            t_starts = []
            t_stops = []
            rates = []
            time_bases = []
            for video_idx, video_timecode in enumerate(video_info[folder_name][block_idx]['start_timestamps']):
                print(f"First video timecode is {video_timecode}")
                video_path = video_info[folder_name][block_idx]['paths'][video_idx]
                rollover = video_info[folder_name][block_idx]['rollovers'][video_idx]
                ## synch based on stream info
                with av.open(video_path) as _file_stream:
                    video_stream = next(s for s in _file_stream.streams if s.type == 'video')
                    if video_stream.average_rate.denominator and video_stream.average_rate.numerator:
                        print(
                            f'video_stream.average_rate = {video_stream.average_rate.numerator} / {video_stream.average_rate.denominator}')
                        fps = float(video_stream.average_rate)
                        time_base = fps ** (-1)
                    elif video_stream.time_base.denominator and video_stream.time_base.numerator:
                        time_base = float(video_stream.time_base)
                        fps = time_base ** (-1)
                    else:
                        raise ValueError("Unable to determine FPS")
                    duration = video_stream.duration * time_base
                    frame_count = video_stream.frames
                rates.append(fps)
                time_bases.append(time_base)
                video_timestamp = timestring_to_timestamp(video_timecode, fps=fps, timecode_type='NDF')
                ripple_timestamp = timestring_to_timestamp(first_ripple['TimeString'], fps=fps, timecode_type='NDF')
                ripple_to_video = (ripple_timestamp - video_timestamp).total_seconds()
                if rollover:
                    # rollover 24 hrs?
                    ripple_to_video += 24 * 60 * 60
                this_t_start = first_ripple['PacketTime'] - ripple_to_video
                t_starts.append(this_t_start)
                these_video_times = np.arange(frame_count) * time_base + this_t_start
                video_times.append(these_video_times)
                this_t_stop = this_t_start + duration
                t_stops.append(this_t_stop)
            video_source = ephyviewer.MultiVideoFileSource(
                video_info[folder_name][block_idx]['paths'], video_times=video_times)
            video_source._t_start = min(t_starts)
            # video_source._t_stop = max(t_stops)
            video_source.t_starts = t_starts
            # video_source.t_stops = t_stops
            video_source.rates = rates
            for video_idx, video_timecode in enumerate(video_info[folder_name][block_idx]['start_timestamps']):
                video_source.frame_grabbers[video_idx].start_time = t_starts[video_idx]
            ## based on offset correction
            '''
            video_source = ephyviewer.MultiVideoFileSource(
                [video_info[folder_name][block_idx]['path']], video_times=None)
            corrected_fps = 29.97
            # pdb.set_trace()
            frame_num = data_dict['ripple']['TimeCode']['TimeString'].apply(count_frames).to_numpy()
            t = data_dict['ripple']['TimeCode']['PacketTime'].to_numpy()
            # fig, ax = plt.subplots()
            # ax.plot(t, frame_num)
            # ax.set_xlabel('Time (sec)')
            # ax.set_ylabel('Frame #')
            # plt.show()
            fps_poly = Polynomial.fit(frame_num, t, deg=1) # coef[0] + coef[1] * x + ...
            corrected_time_base = corrected_fps ** (-1)

            ripple_timestamp = timestring_to_timestamp(first_ripple['TimeString'], fps=corrected_fps, timecode_type='NDF')
            video_timestamp = timestring_to_timestamp(video_timecode, fps=corrected_fps, timecode_type='NDF')
            ripple_to_video = (ripple_timestamp - video_timestamp).total_seconds()
            corrected_t_start = first_ripple['PacketTime'] - ripple_to_video
            #
            video_source._t_start = corrected_t_start
            video_source.t_starts[0] = corrected_t_start
            video_source.frame_grabbers[0].start_time = corrected_t_start
            video_source.frame_grabbers[0].time_base = corrected_time_base
            #
            video_source.frame_grabbers[0].rate = corrected_fps
            video_source.rates[0] = corrected_fps
            '''
        except Exception as e:
            # raise(e)
            pass
        if data_dict['vicon'] is not None:
            if 'EMG' in data_dict['vicon']:
                emg_df = data_dict['vicon']['EMG'].copy()
                emg_df.rename(columns=this_emg_montage, inplace=True)
                emg_df.drop(columns=['NA'], inplace=True)
                emg_sample_rate = np.median(np.diff(emg_df.index.get_level_values('time_usec') * 1e-6)) ** -1
                # filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
                # emg_signals = signal.sosfiltfilt(filterCoeffs, (emg_df - emg_df.mean()).abs(), axis=0)
                emg_signals = (emg_df - emg_df.mean()).abs().to_numpy()
                t_start = emg_df.index.get_level_values('time_usec')[0] * 1e-6
                emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    emg_signals, emg_sample_rate, t_start, channel_names=emg_df.columns)
                emg_signals_view = ephyviewer.TraceViewer(source=emg_signals_source, name=f'block_{block_idx:0>2d}_emg')
                emg_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
                if idx_into_list == 0:
                    win.add_view(emg_signals_view)
                    top_level_emg_view = f'block_{block_idx:0>2d}_emg'
                else:
                    win.add_view(emg_signals_view, tabify_with=top_level_emg_view)
        if data_dict.get('stim_info', None) is not None:
            data_dict['stim_info'].loc[:, 'elecConfig_str'] = data_dict['stim_info'].apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
            # pdb.set_trace()
            data_dict['stim_info'].index = data_dict['stim_info'].index + data_dict['stim_info']['pulseWidth'].to_numpy() * 4
            these_events_list = []
            stim_event_dict = {
                'label': data_dict['stim_info'].apply(
                    lambda x: f'\n{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}', axis='columns').to_numpy(),
                'time': (data_dict['stim_info'].index.get_level_values('timestamp_usec') * 1e-6).to_numpy(),
                'name': f'block_{block_idx:0>2d}_stim_info'
            }
            these_events_list.append(stim_event_dict)
            if True:
                stim_event_original_dict = {
                    'label': data_dict['stim_info'].apply(lambda x: f'\n{x["elecConfig_str"]}\nAmp: {x["amp"]}\nFreq: {x["freq"]}', axis='columns').to_numpy(),
                    'time': (data_dict['stim_info']['original_timestamp_usec'] * 1e-6).to_numpy(),
                    'name': f'block_{block_idx:0>2d}_original_stim_info'
                }
                these_events_list.append(stim_event_original_dict)
            event_source = ephyviewer.InMemoryEventSource(all_events=these_events_list)
            event_view = ephyviewer.EventList(source=event_source, name=f'block_{block_idx:0>2d}_stim_info')
            if idx_into_list == 0:
                win.add_view(event_view, split_with=top_level_emg_view, orientation='horizontal')
                top_level_event_view = f'block_{block_idx:0>2d}_stim_info'
            else:
                win.add_view(event_view, tabify_with=top_level_event_view)
        # pdb.set_trace()
        if 'stim_info_traces' in data_dict:
            if data_dict['stim_info_traces'] is not None:
                stim_info_trace_df = pd.concat(data_dict['stim_info_traces'], names=['feature'], axis='columns')
                stim_info_trace_df.columns = stim_info_trace_df.columns.to_frame().apply(lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
                offset_the_traces = True
                if offset_the_traces:
                    amp_offset = 0
                    amp_step = 20
                    freq_offset = 0
                    freq_step = 100
                    for cn in stim_info_trace_df.columns:
                        if 'amp' in cn:
                            stim_info_trace_df.loc[:, cn] += amp_offset
                            amp_offset += amp_step
                        elif 'freq' in cn:
                            stim_info_trace_df.loc[:, cn] += freq_offset
                            freq_offset += freq_step
                stim_info_trace_sample_rate = np.median(np.diff(stim_info_trace_df.index.get_level_values('time_usec') * 1e-6)) ** -1
                t_start = stim_info_trace_df.index.get_level_values('time_usec')[0] * 1e-6
                stim_info_trace_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                    stim_info_trace_df.to_numpy(), stim_info_trace_sample_rate, t_start, channel_names=stim_info_trace_df.columns)
                stim_info_trace_signals_view = ephyviewer.TraceViewer(source=stim_info_trace_signals_source, name=f'block_{block_idx:0>2d}_stim_info_traces')
                stim_info_trace_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
                if idx_into_list == 0:
                    win.add_view(stim_info_trace_signals_view, split_with=top_level_emg_view, orientation='vertical')
                    top_level_stim_info_trace_view = f'block_{block_idx:0>2d}_stim_info_traces'
                else:
                    win.add_view(stim_info_trace_signals_view, tabify_with=top_level_stim_info_trace_view)
        if data_dict.get('ripple', None) is not None:
            if data_dict['ripple']['TimeCode'] is not None:
                timecode_event_dict = {
                    'label': data_dict['ripple']['TimeCode']['TimeString'].to_numpy(),
                    'time': data_dict['ripple']['TimeCode']['PacketTime'].to_numpy(),
                    'name': f'block_{block_idx:0>2d}_timecode'
                }
                event_source = ephyviewer.InMemoryEventSource(all_events=[timecode_event_dict])
                event_view = ephyviewer.EventList(source=event_source, name=f'block_{block_idx:0>2d}_timecode')
                win.add_view(event_view, tabify_with=f'block_{block_idx:0>2d}_stim_info')
            if data_dict['ripple']['NEV'] is not None:
                nev_spikes_df = data_dict['ripple']['NEV']
                spike_list = []
                # pdb.set_trace()
                for elec, elec_spikes in nev_spikes_df.groupby('Electrode'):
                    if elec < 128:
                        this_label = f"{elec} (caudal)"
                    else:
                        this_label = f"{elec - 128} (rostral)"
                    spike_list.append({
                        'name': this_label,
                        'time': elec_spikes['time_seconds'].to_numpy()
                    })
                spike_source = ephyviewer.InMemorySpikeSource(all_spikes=spike_list)
                spike_view = ephyviewer.SpikeTrainViewer(source=spike_source, name=f'block_{block_idx:0>2d}_nev_spikes')
                spike_view.params_controller.on_automatic_color(cmap_name='Set3')
                if idx_into_list == 0:
                    win.add_view(spike_view, split_with=top_level_emg_view, orientation='vertical')
                    top_level_spike_view = f'block_{block_idx:0>2d}_nev_spikes'
                else:
                    win.add_view(spike_view, tabify_with=top_level_spike_view)
            if 'NF7' in data_dict['ripple']:
                if data_dict['ripple']['NF7'] is not None:
                    # lfp_df = data_dict['ripple']['NF7'].iloc[:, :2].copy()
                    lfp_df = data_dict['ripple']['NF7'].copy()
                    lfp_df.columns = [adjust_channel_name(cn) for cn in lfp_df.columns]
                    del data_dict['ripple']['NF7']
                    gc.collect()
                    lfp_signals = lfp_df.to_numpy()
                    lfp_sample_rate = 15e3
                    t_start = lfp_df.index.get_level_values('time_usec')[0] * 1e-6
                    lfp_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                        lfp_signals, lfp_sample_rate, t_start, channel_names=lfp_df.columns)
                    lfp_signals_view = ephyviewer.TraceViewer(source=lfp_signals_source, name=f'block_{block_idx:0>2d}_lfp')
                    lfp_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
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
                    lfp_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
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
                devices_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
                win.add_view(devices_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
            if ('Points' in data_dict['vicon']):
                points_df = data_dict['vicon']['Points']
                label_mask = points_df.columns.get_level_values('label').str.contains('ForeArm')
                for extra_label in ['Elbow', 'Foot', 'UpperArm', 'Hip', 'Knee', 'Ankle']:
                    label_mask = label_mask | points_df.columns.get_level_values('label').str.contains(extra_label)
                points_df = points_df.loc[:, label_mask].copy()
                angles_dict = {
                    'LeftElbow': ['LeftForeArm', 'LeftElbow', 'LeftUpperArm'],
                    'RightElbow': ['RightForeArm', 'RightElbow', 'RightUpperArm'],
                    'LeftKnee': ['LeftHip', 'LeftKnee', 'LeftAnkle'],
                    'RightKnee': ['RightHip', 'RightKnee', 'RightAnkle'],
                }
                for angle_name, angle_labels in angles_dict.items():
                    try:
                        vec1 = (
                                points_df.xs(angle_labels[0], axis='columns', level='label') -
                                points_df.xs(angle_labels[1], axis='columns', level='label')
                        )
                        vec2 = (
                                points_df.xs(angle_labels[2], axis='columns', level='label') -
                                points_df.xs(angle_labels[1], axis='columns', level='label')
                        )
                        points_df.loc[:, (angle_name, 'angle')] = vg.angle(vec1.to_numpy(), vec2.to_numpy())
                    except Exception as e:
                        pass
                lengths_dict = {
                    'LeftLimb': ['LeftHip', 'LeftFoot'],
                    'RightLimb': ['RightHip', 'RightFoot'],
                }
                for length_name, length_labels in lengths_dict.items():
                    try:
                        vec1 = points_df.xs(length_labels[0], axis='columns', level='label')
                        vec2 = points_df.xs(length_labels[1], axis='columns', level='label')
                        points_df.loc[:, (length_name, 'length')] = vg.euclidean_distance(vec1.to_numpy(), vec2.to_numpy())
                    except Exception as e:
                        pass
                points_df.columns = points_df.columns.to_frame().apply(
                    lambda x: f"{x.iloc[0]}_{x.iloc[1]}", axis=1).to_list()
                points_df.columns.name = 'label'
                # points_df.drop(
                #     columns=[
                #         nm
                #         for nm in points_df.columns
                #         if ('_x' in nm) or ('_y' in nm) or ('_z' in nm)],
                #     inplace=True
                # )
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
                points_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
                win.add_view(points_signals_view, tabify_with=f'block_{block_idx:0>2d}_emg')
            for var_name in ['Accel_X', 'Accel_Y', 'Accel_Z']:
                if var_name in data_dict['vicon']:
                    acc_df = data_dict['vicon'][var_name].copy()
                    acc_df.rename(columns=this_emg_montage, inplace=True)
                    acc_df.drop(columns=['NA'], inplace=True)
                    acc_sample_rate = np.median(np.diff(acc_df.index.get_level_values('time_usec') * 1e-6)) ** -1
                    # filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), acc_sample_rate)
                    # acc_signals = signal.sosfiltfilt(filterCoeffs, acc_df - acc_df.mean(), axis=0)
                    acc_signals = acc_df.to_numpy()
                    t_start = acc_df.index.get_level_values('time_usec')[0] * 1e-6
                    acc_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                        acc_signals, acc_sample_rate, t_start, channel_names=acc_df.columns)
                    acc_signals_view = ephyviewer.TraceViewer(
                        source=acc_signals_source, name=f'block_{block_idx:0>2d}_{var_name}')
                    acc_signals_view.params_controller.on_automatic_color(cmap_name='Set3')
                    win.add_view(acc_signals_view, tabify_with=top_level_emg_view)
                    # if idx_into_list == 0:
                    #     win.add_view(acc_signals_view)
                    #     top_level_acc_view = f'block_{block_idx:0>2d}_{var_name}'
                    # else:
                    #     win.add_view(acc_signals_view, tabify_with=top_level_acc_view)
        try:
            video_view = ephyviewer.VideoViewer(source=video_source, name=f'block_{block_idx:0>2d}_video')
            win.add_view(video_view, tabify_with=f'block_{block_idx:0>2d}_emg')
        except Exception as e:
            print(f'No video found for {folder_name} block {block_idx}')
            # raise(e)
    win.show()
    app.exec_()
    return


if __name__ == '__main__':
    # folder_name = "Day12_PM"
    # list_of_blocks = [4]
    # folder_name = "Day8_AM"
    # list_of_blocks = [4]
    # folder_name = "Day11_AM"
    # list_of_blocks = [4]
    folder_name = "Day11_PM"
    list_of_blocks = [2]
    # folder_name = "Day12_AM"
    # list_of_blocks = [3]
    visualize_dataset(folder_name, list_of_blocks=list_of_blocks)
