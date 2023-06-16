from isicpy.third_party.pymatreader import hdf5todict
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

import pdb, traceback

def sanitize_elec_config(
        elec_cfg):
    if isinstance(elec_cfg, np.float64) or isinstance(elec_cfg, np.float32) or isinstance(elec_cfg, float):
        return [int(elec_cfg), ]
    elif isinstance(elec_cfg, np.ndarray) or isinstance(elec_cfg, list):
        return [int(el) for el in elec_cfg]
    else:
        return None

def sanitize_stim_info(
        si, nev_spikes,
        split_trains=True, force_trains=False,
        calc_rank_in_train=False, remove_zero_amp=True):
    # delta_ts = nev_spikes['TimeStamp'] / 3e4 - nev_spikes['time_seconds']
    # assert np.allclose(delta_ts - delta_ts.iloc[0], 0)
    si.loc[:, 'time'] = ((si['nipTime'] - si['nipTime'].iloc[0]) / 3e4).astype(float) + nev_spikes['time_seconds'].iloc[0]
    for col_name in ['elecCath', 'elecAno']:
        si.loc[:, col_name] = si[col_name].apply(sanitize_elec_config)
    for col_name in ['amp', 'freq', 'pulseWidth', 'res', 'nipTime']:
        si.loc[:, col_name] = si[col_name].astype(np.int64)
    si.loc[:, 'timestamp_usec'] = np.asarray(np.round(si['time'], 6) * 1e6, dtype=np.int64)
    if force_trains:
        si.loc[:, 'isContinuous'] = False
    # align to stim onset
    if remove_zero_amp:
        si = si.loc[si['amp'] != 0, :].reset_index(drop=True)
    si.loc[:, 'original_timestamp_usec'] = si['timestamp_usec'].copy()
    #
    all_spike_times = nev_spikes['time_usec'].copy()
    is_first_spike = (all_spike_times.diff() > 2e5)  # more than 200 msec / 5 Hz
    is_first_spike.iloc[0] = True  # first spike in recording is first in train
    if calc_rank_in_train:
        rank_in_train = is_first_spike.astype(int)
        try:
            for row_idx in nev_spikes.index:
                if not is_first_spike.loc[row_idx]:
                    if row_idx > 0:
                        rank_in_train.loc[row_idx] = rank_in_train.loc[row_idx - 1] + 1
        except Exception:
            traceback.print_exc()
            pdb.set_trace()
        nev_spikes.loc[:, 'rank_in_train'] = rank_in_train
    else:
        nev_spikes.loc[:, 'rank_in_train'] = 0
    first_spike_times = all_spike_times.loc[is_first_spike]
    if split_trains:
        closest_nev_times_train, _ = closestSeries(
            referenceIdx=si.loc[~si['isContinuous'].astype(bool), 'timestamp_usec'],
            sampleFrom=first_spike_times, strictly='neither')
        closest_nev_times_continuous, _ = closestSeries(
            referenceIdx=si.loc[si['isContinuous'].astype(bool), 'timestamp_usec'],
            sampleFrom=all_spike_times, strictly='neither')
        si.loc[si['isContinuous'].astype(bool), 'timestamp_usec'] = closest_nev_times_continuous.to_numpy()
        si.loc[~si['isContinuous'].astype(bool), 'timestamp_usec'] = closest_nev_times_train.to_numpy()
    else:
        closest_nev_times, _ = closestSeries(
            referenceIdx=si['timestamp_usec'],
            sampleFrom=all_spike_times, strictly='neither')
        si.loc[:, 'timestamp_usec'] = closest_nev_times.to_numpy()
    #
    si.loc[:, 'delta_timestamp_usec'] = si['original_timestamp_usec'] - si['timestamp_usec']
    si.set_index('timestamp_usec', inplace=True)
    return si

def closestSeries(
            referenceIdx=None, sampleFrom=None, strictly='neither'):
    closestValues = pd.Series(np.nan, index=referenceIdx.index)
    closestIdx = []
    for idx, value in enumerate(referenceIdx.to_numpy()):
        if strictly == 'greater':
            lookIn = sampleFrom.loc[sampleFrom > value]
        elif strictly == 'less':
            lookIn = sampleFrom.loc[sampleFrom < value]
        else:
            lookIn = sampleFrom
        idxMin = np.abs(lookIn.to_numpy() - value).argmin()
        closeValue = (
            lookIn
            .to_numpy()
            .flat[idxMin])
        closestValues.iloc[idx] = closeValue
        closestIdx.append(lookIn.index[idxMin])
    closestValues = closestValues.astype(referenceIdx.dtype)
    return closestValues, pd.Index(closestIdx)


def convert_nev_electrode_ids(x):
    return 8 - (x - 1) % 8 + 8 * ((x - 1) // 8)

def load_synced_mat(
        base_folder_path=None, session_name=None, block_index=0,
        load_vicon=False, vicon_variable_names=None, vicon_as_df=False, kinematics_time_offset=0,
        interpolate_emg=True, emg_interpolation_cutoff=4e-6,
        load_ripple=False, ripple_variable_names=None, ripple_as_df=False,
        load_stim_info=False, stim_info_variable_names=None, stim_info_as_df=False, split_trains=True, force_trains=False, stim_info_traces=False,
        stim_info_trace_time_vector='EMG',
        load_all_logs=False, all_logs_variable_names=None, all_logs_as_df=False,
        load_meta=False, meta_variable_names=None, meta_as_df=False, verbose=0,
        ):
    ret_dict = {}
    ignore_fields = ['#refs#']
    file_path = base_folder_path / Path(f'3_Preprocessed_Data/{session_name}/Block{block_index:0>4d}_Synced_Session_Data.mat')
    with h5py.File(file_path, 'r') as hdf5_file:
        if load_vicon:
            if verbose > 0:
                print("hdf5todict(hdf5_file['Synced_Session_Data']['Vicon'])")
            ret_dict['vicon'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Vicon'],
                ignore_fields=ignore_fields, variable_names=vicon_variable_names)
            if 'Points' in ret_dict['vicon']:
                ret_dict['vicon']['Points']['Time'] = ret_dict['vicon']['Points']['Time'] + kinematics_time_offset
            for var_name in ['Points', 'Devices', 'EMG', 'Accel_X', 'Accel_Y', 'Accel_Z']:
                if var_name in ret_dict['vicon']:
                    ret_dict['vicon'][var_name]['time_usec'] = np.asarray(
                        np.round(ret_dict['vicon'][var_name]['Time'], 6) * 1e6, dtype=np.int64)
            if vicon_as_df:
                if verbose > 0:
                    print("\tConverting Vicon data to DataFrames")
                if 'Points' in ret_dict['vicon']:
                    x = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][0, :, :].T,
                        index=ret_dict['vicon']['Points']['time_usec'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    y = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][1, :, :].T,
                        index=ret_dict['vicon']['Points']['time_usec'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    z = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][2, :, :].T,
                        index=ret_dict['vicon']['Points']['time_usec'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    df = pd.concat({'x': x, 'y': y, 'z': z}, names=['axis', 'label'], axis='columns')
                    df = df.interpolate().fillna(method='ffill').fillna(method='bfill')
                    df.index.name = 'time_usec'
                    df = df.swaplevel(axis='columns')
                    df.sort_index(axis='columns', inplace=True, level=['label', 'axis'])
                    ret_dict['vicon']['Points'] = df
                if 'Devices' in ret_dict['vicon']:
                    df = pd.DataFrame(
                        ret_dict['vicon']['Devices']['Data'],
                        index=ret_dict['vicon']['Devices']['time_usec']
                        )
                    df.index.name = 'time_usec'
                    df.columns.name = 'label'
                    ret_dict['vicon']['Devices'] = df
                for var_name in ['EMG', 'Accel_X', 'Accel_Y', 'Accel_Z']:
                    if var_name in ret_dict['vicon']:
                        df = pd.DataFrame(
                            ret_dict['vicon'][var_name]['Data'],
                            index=ret_dict['vicon'][var_name]['time_usec']
                            )
                        df.index.name = 'time_usec'
                        df.columns.name = 'label'
                        if interpolate_emg:
                            cutout_mask = df.abs() < emg_interpolation_cutoff
                            for shift_amount in [-1, 1]:
                                cutout_mask = cutout_mask | cutout_mask.shift(shift_amount).fillna(False)
                            df = df.where(~cutout_mask)
                            df.interpolate(inplace=True)
                            df.fillna(0., inplace=True)
                        ret_dict['vicon'][var_name] = df
        if load_ripple:
            if verbose > 0:
                print("hdf5todict(hdf5_file['Synced_Session_Data']['Ripple_Data'])")
            ret_dict['ripple'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Ripple_Data'],
                ignore_fields=ignore_fields, variable_names=ripple_variable_names)
            if 'NEV' in ret_dict['ripple']:
                ret_dict['ripple']['NEV']['Data']['Spikes']['time_usec'] = np.asarray(np.round(
                    ret_dict['ripple']['NEV']['Data']['Spikes']['time_seconds'], 6) * 1e6, dtype=np.int64)
                # sanitize electrode metadata
                for key, list_of_info in ret_dict['ripple']['NEV']['ElectrodesInfo'].items():
                    if key == 'ElectrodeLabel':
                        ret_dict['ripple']['NEV']['ElectrodesInfo'][key] = [
                            None if isinstance(lbl, np.ndarray) else lbl.strip().replace('\x00', '')
                            for lbl in list_of_info
                        ]
                    else:
                        ret_dict['ripple']['NEV']['ElectrodesInfo'][key] = [
                            None if isinstance(lbl, np.ndarray) else lbl
                            for lbl in list_of_info
                        ]
            if 'NS5' in ret_dict['ripple']:
                ret_dict['ripple']['NS5']['time_usec'] = np.asarray(
                    np.round(ret_dict['ripple']['NS5']['time'], 6) * 1e6, dtype=np.int64)
                # sanitize electrode metadata
                for key, list_of_info in ret_dict['ripple']['NS5']['ElectrodesInfo'].items():
                    if key in ['Label', 'AnalogUnits']:
                        ret_dict['ripple']['NS5']['ElectrodesInfo'][key] = [
                            None if isinstance(lbl, np.ndarray) else lbl.strip().replace('\x00', '')
                            for lbl in list_of_info
                        ]
            if 'NF7' in ret_dict['ripple']:
                ret_dict['ripple']['NF7']['time_usec'] = np.asarray(
                    np.round(ret_dict['ripple']['NF7']['time'], 6) * 1e6, dtype=np.int64)
            if 'TimeCode' in ret_dict['ripple']:
                ret_dict['ripple']['TimeCode']['time_usec'] = np.asarray(
                    np.round(ret_dict['ripple']['TimeCode']['PacketTime'], 6) * 1e6, dtype=np.int64)
            if ripple_as_df:
                if verbose > 0:
                    print("\tConverting Ripple data to DataFrames")
                if 'NF7' in ret_dict['ripple']:
                    n_rows = ret_dict['ripple']['NF7']['time_usec'].shape[0]
                    n_cols = len(ret_dict['ripple']['NF7']['AnalogWaveforms'])
                    df = pd.DataFrame(
                        np.zeros((n_rows, n_cols,)),
                        index=ret_dict['ripple']['NF7']['time_usec'],
                        columns=ret_dict['ripple']['NF7']['AnalogEntityLabels'])
                    for idx, data_col in enumerate(ret_dict['ripple']['NF7']['AnalogWaveforms']):
                        df.iloc[:, idx] = data_col
                    df.index.name = 'time_usec'
                    df.columns.name = 'label'
                    ret_dict['ripple']['NF7'] = df
                if 'TimeCode' in ret_dict['ripple']:
                    ret_dict['ripple']['TimeCode'] = pd.DataFrame(ret_dict['ripple']['TimeCode'])
                    ret_dict['ripple']['TimeCode'].set_index('time_usec', inplace=True)
                if 'NS5' in ret_dict['ripple']:
                    column_names = ret_dict['ripple']['NS5']['ElectrodesInfo']['Label']
                    electrodes_info = pd.DataFrame(ret_dict['ripple']['NS5']['ElectrodesInfo'])
                    ret_dict['ripple']['NS5_ElectrodesInfo'] = electrodes_info
                    meta_tags = ret_dict['ripple']['NS5']['MetaTags'].copy()
                    ret_dict['ripple']['NS5_MetaTags'] = meta_tags
                    df = pd.DataFrame(
                        ret_dict['ripple']['NS5']['Data'].T,
                        index=ret_dict['ripple']['NS5']['time_usec'],
                        columns=column_names)
                    df.index.name = 'time_usec'
                    df.columns.name = 'label'
                    ret_dict['ripple']['NS5'] = df
                if 'NEV' in ret_dict['ripple']:
                    meta_tags = ret_dict['ripple']['NEV']['MetaTags'].copy()
                    ret_dict['ripple']['NEV_MetaTags'] = meta_tags
                    electrodes_info = pd.DataFrame(ret_dict['ripple']['NEV']['ElectrodesInfo'])
                    ret_dict['ripple']['NEV_ElectrodesInfo'] = electrodes_info
                    waveform_unit = ret_dict['ripple']['NEV']['Data']['Spikes'].pop('WaveformUnit')
                    waveform_df = pd.DataFrame(ret_dict['ripple']['NEV']['Data']['Spikes'].pop('Waveform'))
                    elec_ids = ret_dict['ripple']['NEV']['Data']['Spikes']['Electrode']
                    elec_ids[elec_ids > 5120] = elec_ids[elec_ids > 5120] - 5120
                    ret_dict['ripple']['NEV'] = pd.DataFrame(ret_dict['ripple']['NEV']['Data']['Spikes'])
                    ret_dict['ripple']['NEV'].loc[:, 'Electrode'] = ret_dict['ripple']['NEV']['Electrode'].apply(convert_nev_electrode_ids)
        if load_meta:
            ret_dict['meta'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Meta'],
                ignore_fields=ignore_fields, variable_names=meta_variable_names)
    if load_stim_info:
        logs_folder = base_folder_path / Path(f'1_Raw_Data/{session_name}')
        stim_log_paths = [path for path in Path(logs_folder).rglob(f'*Block{block_index:0>4d}*autoStimLog.json')]
        assert len(stim_log_paths) == 1
        json_path = base_folder_path / stim_log_paths[0]
        raw_stim_info = pd.read_json(json_path, orient='records').iloc[:, 0]
        raw_stim_info = pd.concat([pd.Series(d) for d in raw_stim_info], axis='columns').T
        '''
        all_log_paths = [path for path in Path(logs_folder).rglob(f'*Block{block_index:0>4d}*allLogs.json')]
        assert len(all_log_paths) == 1
        all_json_path = base_folder_path / all_log_paths[0]
        all_logs = pd.read_json(all_json_path, orient='records')
        all_logs.loc[:, 'datetime'] =  all_logs.apply(lambda x: pd.Timestamp(f"{x['date']} {x['day-time']}", tz='US/Eastern'), axis='columns')
        all_logs_nip_time = all_logs.loc[all_logs['msg_type'] == 'stim_event_json', 'msg'].apply(lambda x: pd.Timedelta(seconds = x['nipTime'] / 3e4))
        all_logs_datetime = all_logs.loc[all_logs['msg_type'] == 'stim_event_json', 'datetime']
        nip_on_time = (all_logs_datetime - all_logs_nip_time).mean()
        '''
        with h5py.File(file_path, 'r') as hdf5_file:
            just_nev = hdf5todict(
                hdf5_file['Synced_Session_Data']['Ripple_Data'],
                ignore_fields=ignore_fields, variable_names=['NEV'])
            # yr, mo, _, da, hr, mi, se, ms  = just_nev['NEV']['MetaTags']['DateTimeRaw']
            # nev_rec_datetime = pd.Timestamp(year=int(yr), month=int(mo), day=int(da), hour=int(hr), minute=int(mi), second=int(se), microsecond=int(ms * 1e3), tz='UTC')
            just_nev['NEV']['Data']['Spikes']['time_usec'] = np.asarray(
                np.round(just_nev['NEV']['Data']['Spikes']['time_seconds'], 6) * 1e6, dtype=np.int64)
            waveform_unit = just_nev['NEV']['Data']['Spikes'].pop('WaveformUnit')
            waveform_df = pd.DataFrame(just_nev['NEV']['Data']['Spikes'].pop('Waveform'))
            nev_spikes = pd.DataFrame(just_nev['NEV']['Data']['Spikes'])
            # nev_spikes.loc[:, 'TimeStamp'] = nev_spikes['TimeStamp'] + int((nev_rec_datetime - nip_on_time).total_seconds() * 3e4)
        if verbose > 0:
            print("\tret_dict['stim_info'] = sanitize_stim_info(...")
        ret_dict['stim_info'] = sanitize_stim_info(
            raw_stim_info, nev_spikes,
            split_trains=split_trains, force_trains=force_trains)
        if stim_info_traces:
            if verbose > 0:
                print("\tfull_stim_info = sanitize_stim_info(...")
            full_stim_info = sanitize_stim_info(
                raw_stim_info, nev_spikes,
                split_trains=split_trains,
                force_trains=force_trains, remove_zero_amp=False)
            full_stim_info.loc[:, 'elecConfig_str'] = full_stim_info.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
            config_metadata = full_stim_info.loc[:, ['elecCath', 'elecAno', 'elecConfig_str']].drop_duplicates().set_index('elecConfig_str')
            time_vector = ret_dict['vicon']['EMG'].index  # in usec
            stim_info_amplitude = pd.DataFrame(np.nan, index=time_vector, columns=config_metadata.index)
            stim_info_amplitude.iloc[0, :] = 0
            if '-[0]+[0]' in stim_info_amplitude.columns:
                stim_info_amplitude.drop(columns='-[0]+[0]', inplace=True)
            if '-[]+[]' in stim_info_amplitude.columns:
                stim_info_amplitude.drop(columns='-[]+[]', inplace=True)
            stim_info_freq = pd.DataFrame(np.nan, index=time_vector, columns=config_metadata.index)
            stim_info_freq.iloc[0, :] = 0
            if '-[0]+[0]' in stim_info_freq.columns:
                stim_info_freq.drop(columns='-[0]+[0]', inplace=True)
            if '-[]+[]' in stim_info_freq.columns:
                stim_info_freq.drop(columns='-[]+[]', inplace=True)
            if verbose > 0:
                print("\tEntering stim_traces loop")
            for _, stim_group in full_stim_info.reset_index().groupby('timestamp_usec'):
                time_usec = stim_group['timestamp_usec'].iloc[0]
                time_mask = stim_info_amplitude.index > time_usec
                if time_mask.any():
                    nearest_time = stim_info_amplitude.index[time_mask][0]
                    stim_info_amplitude.loc[nearest_time, :] = 0
                    stim_info_freq.loc[nearest_time, :] = 0
                    for _, this_stim_update in stim_group.iterrows():
                        this_electrode_label = this_stim_update['elecConfig_str']
                        # if this_electrode_label not in ('-[0]+[0]', '-[]+[]',):
                        stim_info_amplitude.loc[nearest_time, this_electrode_label] = this_stim_update['amp']
                        stim_info_freq.loc[nearest_time, this_electrode_label] = this_stim_update['freq']
            stim_info_freq = stim_info_freq.fillna(method='ffill').fillna(method='bfill')
            stim_info_amplitude = stim_info_amplitude.fillna(method='ffill').fillna(method='bfill')
            stim_info_amplitude.clip(lower=-15000, upper=15000, inplace=True)
            ret_dict['stim_info_traces'] = {
                'amp': stim_info_amplitude, 'freq': stim_info_freq
                }
    if load_all_logs:
        ret_dict['all_logs'] = all_logs
    return ret_dict
