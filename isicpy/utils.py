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
# function to load table variable from MAT-file
# based on https://stackoverflow.com/questions/25853840/load-matlab-tables-in-python-using-scipy-io-loadmat
def loadtablefrommat(
        mat):
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
        si, nev_spikes, split_trains=True, force_trains=False, remove_zero_amp=True):
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
    first_spike_times = all_spike_times.loc[is_first_spike]
    if split_trains:
        closest_nev_times_train, _ = closestSeries(
            referenceIdx=si.loc[si['isContinuous'].astype(bool), 'timestamp_usec'],
            sampleFrom=first_spike_times, strictly='neither')
        closest_nev_times_continuous, _ = closestSeries(
            referenceIdx=si.loc[~si['isContinuous'].astype(bool), 'timestamp_usec'],
            sampleFrom=all_spike_times, strictly='neither')
        si.loc[si['isContinuous'].astype(bool), 'timestamp_usec'] = closest_nev_times_train.to_numpy()
        si.loc[~si['isContinuous'].astype(bool), 'timestamp_usec'] = closest_nev_times_continuous.to_numpy()
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


def load_synced_mat(
        file_path=None,
        load_vicon=False, vicon_variable_names=None, vicon_as_df=False,
        interpolate_emg=True,
        load_ripple=False, ripple_variable_names=None, ripple_as_df=False,
        load_stim_info=False, stim_info_variable_names=None, stim_info_as_df=False, split_trains=True, force_trains=False, stim_info_traces=True,
        stim_info_trace_time_vector='EMG',
        load_all_logs=False, all_logs_variable_names=None, all_logs_as_df=False,
        load_meta=False, meta_variable_names=None, meta_as_df=False,
        ):
    ret_dict = {}
    ignore_fields = ['#refs#']
    with h5py.File(file_path, 'r') as hdf5_file:
        # hdf5_file = h5py.File(file_path, 'r')
        if load_vicon:
            ret_dict['vicon'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Vicon'],
                ignore_fields=ignore_fields, variable_names=vicon_variable_names)
            if 'Points' in ret_dict['vicon']:
                ret_dict['vicon']['Points']['time_usec'] = np.asarray(np.round(ret_dict['vicon']['Points']['Time'], 6) * 1e6, dtype=np.int64)
            if 'Devices' in ret_dict['vicon']:
                ret_dict['vicon']['Devices']['time_usec'] = np.asarray(np.round(ret_dict['vicon']['Devices']['Time'], 6) * 1e6, dtype=np.int64)
            if 'EMG' in ret_dict['vicon']:
                ret_dict['vicon']['EMG']['time_usec'] = np.asarray(np.round(ret_dict['vicon']['EMG']['Time'], 6) * 1e6, dtype=np.int64)
            if vicon_as_df:
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
                if 'EMG' in ret_dict['vicon']:
                    df = pd.DataFrame(
                        ret_dict['vicon']['EMG']['Data'],
                        index=ret_dict['vicon']['EMG']['time_usec']
                        )
                    df.index.name = 'time_usec'
                    df.columns.name = 'label'
                    if interpolate_emg:
                        '''
                            filterOpts = {
                                'low': {
                                    'Wn': 500.,
                                    'N': 4,
                                    'btype': 'low',
                                    'ftype': 'butter'
                                },
                            }
                            analog_time_vector = np.asarray(df.index)
                            nominal_dt = np.float64(np.median(np.diff(analog_time_vector)))
                            emg_sample_rate = (nominal_dt ** -1) * 1e6
                            filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
                            filtered_df = pd.DataFrame(signal.sosfiltfilt(filterCoeffs, df, axis=0), index=df.index, columns=df.columns)
                            df = filtered_df
                        '''
                        #  sample_data = df.iloc[:, [0,1]].abs().unstack().to_numpy()
                        #  bins = np.linspace(0, 1e-6, 100).tolist() + np.linspace(1e-6, sample_data.max(), 100).tolist()
                        #  plt.hist(sample_data, bins=bins); plt.show()
                        cutout_mask = df.abs() < 2.5e-7
                        for shift_amount in [-1, 1]:
                            cutout_mask = cutout_mask | cutout_mask.shift(shift_amount).fillna(False)
                        df = df.where(~cutout_mask)
                        df.interpolate(inplace=True)
                        df.fillna(0., inplace=True)
                    ret_dict['vicon']['EMG'] = df
        if load_ripple:
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
            if ripple_as_df:
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
        if load_stim_info:
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
            ret_dict['stim_info'] = sanitize_stim_info(
                loadtablefrommat(ret_dict['stim_info']), nev_spikes, split_trains=split_trains, force_trains=force_trains)
            if stim_info_traces:
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
                stim_info_freq = pd.DataFrame(np.nan, index=time_vector, columns=config_metadata.index)
                stim_info_freq.iloc[0, :] = 0
                if '-(0,)+(0,)' in stim_info_freq.columns:
                    stim_info_freq.drop(columns='-(0,)+(0,)', inplace=True)
                for time_usec, this_stim_update in tqdm(full_stim_info.iterrows(), total=full_stim_info.shape[0]):
                    time_mask = stim_info_amplitude.index > time_usec
                    if time_mask.any():
                        nearest_time = stim_info_amplitude.index[time_mask][0]
                        this_electrode_label = this_stim_update['elecConfig_str']
                        stim_info_amplitude.loc[nearest_time, this_electrode_label] = this_stim_update['amp']
                        stim_info_freq.loc[nearest_time, this_electrode_label] = this_stim_update['freq']
                        # stim_info_amplitude.loc[nearest_time, :] = 0
                        # stim_info_freq.loc[nearest_time, :] = 0
                        # if this_electrode_label != '-(0,)+(0,)':
                        #     stim_info_amplitude.loc[nearest_time, this_electrode_label] = this_stim_update['amp']
                        #     stim_info_freq.loc[nearest_time, this_electrode_label] = this_stim_update['freq']
                stim_info_freq = stim_info_freq.fillna(method='ffill').fillna(method='bfill')
                stim_info_amplitude = stim_info_amplitude.fillna(method='ffill').fillna(method='bfill')
                ret_dict['stim_info_traces'] = {
                    'amp': stim_info_amplitude,
                    'freq': stim_info_freq
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
                bw = w0/notchQ
                theseOpts['Wn'] = [w0 - bw/2, w0 + bw/2]
                sos = signal.iirfilter(
                        **theseOpts, output='sos')
                filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
                print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
                if plotting:
                    plotFilterOptsResponse(theseOpts)
        if theseOpts['btype'] == 'high':
            theseOpts['fs'] = samplingRate
            sos = signal.iirfilter(
                    **theseOpts, output='sos')
            filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
            print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
            if plotting:
                plotFilterOptsResponse(theseOpts)
        #
        if theseOpts['btype'] == 'low':
            theseOpts['fs'] = samplingRate
            sos = signal.iirfilter(
                **theseOpts, output='sos')
            filterCoeffsSOS = np.concatenate([filterCoeffsSOS, sos])
            print('Adding {} coefficients for filter portion {}'.format(sos.shape[0], fName))
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
