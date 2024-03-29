import pandas as pd
import numpy as np
from isicpy.third_party.pymatreader import hdf5todict
import h5py
import pdb
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import traceback
from collections.abc import Iterable

# function to load table variable from MAT-file
# based on https://stackoverflow.com/questions/25853840/load-matlab-tables-in-python-using-scipy-io-loadmat
def loadtablefrommat(mat):
    """
    read a struct-ified table variable (and column names) from a MAT-file
    and return pandas.DataFrame object.
    """
    # get table (struct) variable
    if len(mat):
        data = mat['table']['data']
    else:
        return None
    # create dict out of original table
    table_dict = {
        mat['columns'][idx]: datum
        for idx, datum in enumerate(mat['table']['data'])
        }
    return pd.DataFrame(table_dict)


def sanitize_elec_config(
        elec_cfg):
    if isinstance(elec_cfg, np.float64) or isinstance(elec_cfg, np.float32) or isinstance(elec_cfg, float):
        return (int(elec_cfg), )
    elif isinstance(elec_cfg, np.ndarray):
        # print(tuple(int(el) for el in elec_cfg))
        return tuple(int(el) for el in elec_cfg)
    else:
        pdb.set_trace()


def sanitize_stim_info(
        si, nev_spikes,
        split_trains=True, force_trains=False,
        calc_rank_in_train=False, remove_zero_amp=True):
    ####################################################################################################################
    # handle updatse applied to multiple electrode configurations simultaneously, i.e. rows that reference multiple
    # amplitudes
    indexes_to_replace = []
    new_entries = []
    for row_idx, row in si.iterrows():
        if isinstance(row['amp'], Iterable):
            indexes_to_replace.append(row_idx)
            for idx in range(len(row['amp'])):
                new_row = row.copy()
                for col_name in ['elecCath', 'elecAno', 'amp', 'freq', 'pulseWidth', 'isContinuous']:
                    new_row.loc[col_name] = row[col_name][idx]
                new_entries.append(new_row)
    if len(new_entries):
        new_entries_df = pd.concat(new_entries, axis='columns').T
        si.drop(index=indexes_to_replace, inplace=True)
        si = pd.concat([si, new_entries_df], ignore_index=True)
        si.sort_values(by='nipTime', inplace=True)
    ####################################################################################################################
    for col_name in ['elecCath', 'elecAno']:
        si.loc[:, col_name] = si[col_name].apply(sanitize_elec_config)
    for col_name in ['amp', 'freq', 'pulseWidth', 'res', 'nipTime']:
        si.loc[:, col_name] = si[col_name].astype(np.int64)
    si.loc[:, 'timestamp_usec'] = np.asarray(np.round(si['time'].astype(float), 6) * 1e6, dtype=np.int64)
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


def sanitize_all_logs(al):
    pdb.set_trace()


def convert_nev_electrode_ids(x):
    return 8 - (x - 1) % 8 + 8 * ((x - 1) // 8)

def load_synced_mat(
        file_path=None,
        load_vicon=False, vicon_variable_names=None, vicon_as_df=False, kinematics_time_offset=0,
        interpolate_emg=True,
        load_ripple=False, ripple_variable_names=None, ripple_as_df=False,
        load_stim_info=False, stim_info_variable_names=None, stim_info_as_df=False, split_trains=True, force_trains=False, stim_info_traces=False,
        stim_info_trace_time_vector='EMG',
        load_all_logs=False, all_logs_variable_names=None, all_logs_as_df=False,
        load_meta=False, meta_variable_names=None, meta_as_df=False, verbose=0,
        ):
    ret_dict = {}
    ignore_fields = ['#refs#']
    with h5py.File(file_path, 'r') as hdf5_file:
        # hdf5_file = h5py.File(file_path, 'r')
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
                    ret_dict['vicon'][var_name]['time_usec'] = np.asarray(np.round(ret_dict['vicon'][var_name]['Time'], 6) * 1e6, dtype=np.int64)
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
                            cutout_mask = df.abs() < 4e-6
                            for shift_amount in [-1, 1]:
                                cutout_mask = cutout_mask | cutout_mask.shift(shift_amount).fillna(False)
                            # pdb.set_trace()
                            '''
                            plot_mask = (df.index > 0) & (df.index < 1e6)
                            plt.plot(df.iloc[plot_mask, 6])
                            plt.plot(df.iloc[(plot_mask & cutout_mask.iloc[:, 6]).to_numpy(), 6], 'r*')
                            plt.show()
                            '''
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
                ret_dict['ripple']['NEV']['Data']['Spikes']['time_usec'] = np.asarray(np.round(ret_dict['ripple']['NEV']['Data']['Spikes']['time_seconds'], 6) * 1e6, dtype=np.int64)
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
        if load_stim_info:
            if verbose > 0:
                print("hdf5todict(hdf5_file['Synced_Session_Data']['StimInfo'])")
            ret_dict['stim_info'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['StimInfo'],
                ignore_fields=ignore_fields, variable_names=stim_info_variable_names)
            just_nev = hdf5todict(
                hdf5_file['Synced_Session_Data']['Ripple_Data'],
                ignore_fields=ignore_fields, variable_names=['NEV'])
            just_nev['NEV']['Data']['Spikes']['time_usec'] = np.asarray(
                np.round(just_nev['NEV']['Data']['Spikes']['time_seconds'], 6) * 1e6, dtype=np.int64)
            waveform_unit = just_nev['NEV']['Data']['Spikes'].pop('WaveformUnit')
            waveform_df = pd.DataFrame(just_nev['NEV']['Data']['Spikes'].pop('Waveform'))
            nev_spikes = pd.DataFrame(just_nev['NEV']['Data']['Spikes'])
            if verbose > 0:
                print("\tret_dict['stim_info'] = sanitize_stim_info(loadtablefrommat(ret_dict['stim_info']), nev_spikes,...")
            ret_dict['stim_info'] = sanitize_stim_info(
                loadtablefrommat(ret_dict['stim_info']), nev_spikes,
                split_trains=split_trains, force_trains=force_trains
                )
            if stim_info_traces:
                if verbose > 0:
                    print("hdf5todict(hdf5_file['Synced_Session_Data']['StimInfo'])\t(for traces)")
                full_stim_info = hdf5todict(
                    hdf5_file['Synced_Session_Data']['StimInfo'],
                    ignore_fields=ignore_fields, variable_names=stim_info_variable_names)
                full_stim_info = sanitize_stim_info(
                    loadtablefrommat(full_stim_info), nev_spikes,
                    split_trains=split_trains,
                    force_trains=force_trains, remove_zero_amp=False)
                full_stim_info.loc[:, 'elecConfig_str'] = full_stim_info.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
                config_metadata = full_stim_info.loc[:, ['elecCath', 'elecAno', 'elecConfig_str']].drop_duplicates().set_index('elecConfig_str')
                time_vector = ret_dict['vicon']['EMG'].index  # in usec
                #
                stim_info_amplitude = pd.DataFrame(np.nan, index=time_vector, columns=config_metadata.index)
                stim_info_amplitude.iloc[0, :] = 0
                if '-(0,)+(0,)' in stim_info_amplitude.columns:
                    stim_info_amplitude.drop(columns='-(0,)+(0,)', inplace=True)
                if '-()+()' in stim_info_amplitude.columns:
                    stim_info_amplitude.drop(columns='-()+()', inplace=True)
                stim_info_freq = pd.DataFrame(np.nan, index=time_vector, columns=config_metadata.index)
                stim_info_freq.iloc[0, :] = 0
                if '-(0,)+(0,)' in stim_info_freq.columns:
                    stim_info_freq.drop(columns='-(0,)+(0,)', inplace=True)
                if '-()+()' in stim_info_freq.columns:
                    stim_info_freq.drop(columns='-()+()', inplace=True)
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
                            # if this_electrode_label not in ('-(0,)+(0,)', '-()+()',):
                            stim_info_amplitude.loc[nearest_time, this_electrode_label] = this_stim_update['amp']
                            stim_info_freq.loc[nearest_time, this_electrode_label] = this_stim_update['freq']
                stim_info_freq = stim_info_freq.fillna(method='ffill').fillna(method='bfill')
                stim_info_amplitude = stim_info_amplitude.fillna(method='ffill').fillna(method='bfill')
                stim_info_amplitude.clip(lower=-15000, upper=15000, inplace=True)
                ret_dict['stim_info_traces'] = {
                    'amp': stim_info_amplitude, 'freq': stim_info_freq
                    }
        if load_all_logs:
            ret_dict['all_logs'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['AllLogs'],
                ignore_fields=ignore_fields, variable_names=all_logs_variable_names)
            ret_dict['all_logs'] = loadtablefrommat(ret_dict['all_logs'])
        if load_meta:
            ret_dict['meta'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Meta'],
                ignore_fields=ignore_fields, variable_names=meta_variable_names)
    return ret_dict


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


def applySavGol(
        df, window_length_sec=None,
        polyorder=None, fs=None,
        deriv=0, pos=None, columns=None):
    if fs is None:
        fs = 1.
    delta = float(fs) ** (-1)
    window_length = int(2 * np.ceil(fs * window_length_sec / 2) + 1)
    polyorder = min(polyorder, window_length - 1)
    passedSeries = False
    if isinstance(df, pd.Series):
        passedSeries = True
        data = df.to_frame(name='temp')
        columns = ['temp']
    else:
        data = df.copy()
    if columns is None:
        columns = df.columns
    savGolCoeffs = signal.savgol_coeffs(
        window_length, polyorder, deriv=deriv,
        delta=delta, pos=pos)
    for cName in columns:
        data.loc[:, cName] = np.convolve(data[cName], savGolCoeffs, mode='same')
    if passedSeries:
        data = data['temp']
    return data


def makeFilterCoeffsSOS(
        filterOpts, samplingRate, plotting=False):
    fOpts = deepcopy(filterOpts)
    filterCoeffsSOS = np.ndarray(shape=(0, 6))
    #
    for fName, theseOpts in fOpts.items():
        if theseOpts['btype'] == 'bandstop':
            nNotchHarmonics = theseOpts.pop('nHarmonics')
            notchFreq = theseOpts.pop('Wn')
            notchQ = theseOpts.pop('Q')
            theseOpts['fs'] = samplingRate
            for harmonicOrder in range(1, nNotchHarmonics + 1):
                w0 = harmonicOrder * notchFreq
                bw = w0 / notchQ
                theseOpts['Wn'] = [w0 - bw/2, w0 + bw/2]
                sos = signal.iirfilter(**theseOpts, output='sos')
                filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
                # print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
                if plotting:
                    plotFilterOptsResponse(theseOpts)
        if theseOpts['btype'] == 'high':
            theseOpts['fs'] = samplingRate
            sos = signal.iirfilter(**theseOpts, output='sos')
            filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
            # print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
            if plotting:
                plotFilterOptsResponse(theseOpts)
        #
        if theseOpts['btype'] == 'low':
            theseOpts['fs'] = samplingRate
            sos = signal.iirfilter(**theseOpts, output='sos')
            filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
            # print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
            if plotting:
                plotFilterOptsResponse(theseOpts)
    return filterCoeffsSOS


def plotFilterOptsResponse(filterOpts):
    sos = signal.iirfilter(**filterOpts, output='sos')
    fig, ax1, ax2 = plotFilterResponse(sos, filterOpts['fs'])
    ax1.set_title('{}'.format(filterOpts['btype']))
    if isinstance(filterOpts['Wn'], list):
        for Wn in filterOpts['Wn']:
            ax1.axvline(
                Wn, color='green', linestyle='--')  # cutoff frequency
    else:
        ax1.axvline(
            filterOpts['Wn'],
            color='green', linestyle='--')  # cutoff frequency
    plt.show()
    return


def plotFilterResponse(sos, fs):
    w, h = signal.sosfreqz(sos, worN=2048, fs=fs)
    angles = np.unwrap(np.angle(h))
    fig, ax1 = plt.subplots()
    ax1.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude [dB]')
    ax2 = ax1.twinx()
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    return fig, ax1, ax2


def plotFilterImpulseResponse(
        fOpts, fs, useAcausal=True):
    fig2, ax3 = plt.subplots()
    ax4 = ax3.twinx()
    fCoeffs = makeFilterCoeffsSOS(
        fOpts, fs)
    nMult = 3
    if 'high' in fOpts:
        ti = np.arange(
            -nMult * fOpts['high']['Wn'] ** (-1),
            nMult * fOpts['high']['Wn'] ** (-1),
            fs ** (-1))
    else:
        ti = np.arange(-.2, .2, fs ** (-1))
    impulse = np.zeros(ti.shape)
    impulse[int(ti.shape[0]/2)] = 1
    if useAcausal:
        filtered = signal.sosfiltfilt(fCoeffs, impulse)
    else:
        filtered = signal.sosfilt(fCoeffs, impulse)
    ax3.plot(ti, impulse, c='tab:blue', label='impulse')
    ax3.legend(loc='upper right')
    ax4.plot(ti, filtered, c='tab:orange', label='filtered')
    ax4.legend(loc='lower right')
    return fig2, ax3, ax4


def count_frames(tcode_str):
    frame_hour, frame_minute, frame_second, frame = [int(num) for num in tcode_str.split(':')]
    total_frames = frame_hour * (60 ** 2) * 30 + frame_minute * 60 * 30 + frame_second * 30 + frame
    return total_frames

def timestring_to_timestamp(
        tcode, fps=29.97,
        year=2022, month=10, day=31,
        timecode_type='DF'):
    if timecode_type == 'DF':
        hour, minute, second, frame = [int(num) for num in tcode.split(':')]
        usec = int(1e6 * frame / fps)
        tstamp = pd.Timestamp(
            year=year, month=month, day=day,
            hour=hour, minute=minute, second=second,
            microsecond=usec)
    elif timecode_type == 'NDF':
        total_frames = count_frames(tcode)
        t_delta = pd.Timedelta(total_frames / fps, unit='sec')
        tstamp = pd.Timestamp(year=year, month=month, day=day) + t_delta
    return tstamp

def confirmTriggersPlot(peakIdx, dataSeries, fs, whichPeak=0, nSec=10):
    #
    indent = peakIdx[whichPeak]
    #
    dataSlice = slice(
        max(0, int(indent-.25*fs)),
        min(int(indent+nSec*fs), dataSeries.shape[0])) # 5 sec after first peak
    peakSlice = np.where(np.logical_and(peakIdx > indent - .25*fs, peakIdx < indent + nSec*fs))
    #
    fig, ax = plt.subplots()
    plt.plot((dataSeries.index[dataSlice] - indent) * fs ** (-1), dataSeries.iloc[dataSlice])
    plt.plot((peakIdx[peakSlice] - indent) * fs ** (-1), dataSeries.iloc[peakIdx[peakSlice]], 'r*')
    plt.title('dataSeries and found triggers')
    plt.xlabel('distance between triggers (sec)')

    figDist, axDist = plt.subplots()
    if len(peakIdx) > 5:
        sns.distplot(np.diff(peakIdx) * fs ** (-1), kde=False)
    plt.title('distance between triggers (sec)')
    plt.xlabel('distance between triggers (sec)')
    return fig, ax, figDist, axDist

def getThresholdCrossings(
        dataSrs, thresh=None, absVal=False,
        edgeType='rising', fs=3e4, iti=None,
        keep_max=True, itiWiggle=0.05,
        plotting=False, plot_opts=dict()):
    if absVal:
        dsToSearch = dataSrs.abs()
    else:
        dsToSearch = dataSrs
    # dsToSearch: data series to search
    nextDS = dsToSearch.shift(1).fillna(method='bfill')
    if edgeType == 'rising':
        crossMask = (
            (dsToSearch >= thresh) & (nextDS < thresh) |
            (dsToSearch > thresh) & (nextDS <= thresh))
    elif edgeType == 'falling':
        crossMask = (
            (dsToSearch <= thresh) & (nextDS > thresh) |
            (dsToSearch < thresh) & (nextDS >= thresh))
    elif edgeType == 'both':
        risingMask = (
            (dsToSearch >= thresh) & (nextDS < thresh) |
            (dsToSearch > thresh) & (nextDS <= thresh))
        fallingMask = (
            (dsToSearch <= thresh) & (nextDS > thresh) |
            (dsToSearch < thresh) & (nextDS >= thresh))
        crossMask = risingMask | fallingMask
    crossIdx = dataSrs.index[crossMask]
    if iti is not None:
        min_dist = int(fs * iti * (1 - itiWiggle))
        y = dsToSearch.abs().to_numpy()
        peaks = np.array([dsToSearch.index.get_loc(i) for i in crossIdx])
        if peaks.size > 1 and min_dist > 1:
            if keep_max:
                highest = peaks[np.argsort(y[peaks])][::-1]
            else:
                highest = peaks
            rem = np.ones(y.size, dtype=bool)
            rem[peaks] = False
            for peak in highest:
                if not rem[peak]:
                    sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                    rem[sl] = True
                    rem[peak] = False
            peaks = np.arange(y.size)[~rem]
            crossIdx = dsToSearch.index[peaks]
            crossMask = dsToSearch.index.isin(crossIdx)
    if plotting and (crossIdx.size > 0):
        figData, axData, figDist, axDist = confirmTriggersPlot(crossIdx, dsToSearch, fs, **plot_opts)
        plt.show(block=True)
    return crossIdx, crossMask


def mapToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='; ',
        skiprows=10, header=None, engine='python',
        names=['FE', 'electrode', 'position'])
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'zcoords', 'elecName',
            'elecID', 'label', 'bank', 'bankID', 'nevID']
        )
    bankLookup = {
        'A.1': 0, 'A.2': 1, 'A.3': 2, 'A.4': 3,
        'B.1': 4, 'B.2': 5, 'B.3': 6, 'B.4': 7}
    for rowIdx, row in arrayMap.iterrows():
        processor, port, FEslot, channel = row['FE'].split('.')
        bankName = '{}.{}'.format(port, FEslot)
        array, electrodeFull = row['electrode'].split('.')
        if '_' in electrodeFull:
            electrode, electrodeRep = electrodeFull.split('_')
        else:
            electrode = electrodeFull
        x, y, z = row['position'].split('.')
        nevIdx = int(channel) - 1 + bankLookup[bankName] * 32
        cmpDF.loc[nevIdx, 'elecID'] = int(electrode[1:])
        cmpDF.loc[nevIdx, 'nevID'] = nevIdx
        cmpDF.loc[nevIdx, 'elecName'] = array
        cmpDF.loc[nevIdx, 'xcoords'] = float(x)
        cmpDF.loc[nevIdx, 'ycoords'] = float(y)
        cmpDF.loc[nevIdx, 'zcoords'] = float(z)
        cmpDF.loc[nevIdx, 'label'] = row['electrode'].replace('.', '_')
        cmpDF.loc[nevIdx, 'bank'] = bankName
        cmpDF.loc[nevIdx, 'bankID'] = int(channel)
        cmpDF.loc[nevIdx, 'FE'] = row['FE']
    #
    cmpDF.dropna(inplace=True)
    cmpDF.reset_index(inplace=True, drop=True)
    cmpDF.loc[:, 'nevID'] += 1
    return cmpDF

