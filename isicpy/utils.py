import pandas as pd
import numpy as np
from isicpy.third_party.pymatreader import hdf5todict
import h5py
import pdb

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

def sanitize_elec_config(elec_cfg):
    if isinstance(elec_cfg, np.float64) or isinstance(elec_cfg, np.float32) or isinstance(elec_cfg, float):
        return (int(elec_cfg), )
    elif isinstance(elec_cfg, np.ndarray):
        print((int(el) for el in elec_cfg))
        return (int(el) for el in elec_cfg)
    else:
        pdb.set_trace()


def sanitize_stim_info(si, nev_spikes):
    for col_name in ['elecCath', 'elecAno']:
        si.loc[:, col_name] = si[col_name].apply(sanitize_elec_config)
    for col_name in ['amp', 'freq', 'pulseWidth', 'res', 'nipTime']:
        si.loc[:, col_name] = si[col_name].astype(np.int64)
    si.loc[:, 'timestamp_usec'] = np.asarray(np.round(si['time'], 6) * 1e6, dtype=np.int64)

    # align to stim onset
    si = si.loc[si['amp'] != 0, :].reset_index(drop=True)
    #
    all_spike_times = nev_spikes['time_usec']
    is_first_spike = (all_spike_times.diff() > 2e5)  # more than 200 msec / 5 Hz
    is_first_spike.iloc[0] = True  # first spike in recording is first in train
    rank_in_train = is_first_spike.astype(int)
    for row_idx in nev_spikes.index:
        if not is_first_spike.loc[row_idx]:
            if row_idx > 0:
                rank_in_train.loc[row_idx] = rank_in_train.loc[row_idx - 1] + 1
    nev_spikes.loc['rank_in_train'] = rank_in_train
    first_spike_times = all_spike_times.loc[is_first_spike]
    closest_nev_times, _ = closestSeries(
        referenceIdx=si['timestamp_usec'],
        sampleFrom=first_spike_times, strictly='neither')
    #
    si.loc[:, 'original_timestamp_usec'] = si['timestamp_usec'].copy()
    si.loc[:, 'timestamp_usec'] = closest_nev_times.to_numpy()
    si.loc[:, 'delta_timestamp_usec'] = si['original_timestamp_usec'].to_numpy() - closest_nev_times.to_numpy()
    si.set_index('timestamp_usec', inplace=True)
    return si

def sanitize_all_logs(al):
    pdb.set_trace()
def load_synced_mat(
        file_path=None,
        load_vicon=False, vicon_variable_names=None, vicon_as_df=False,
        load_ripple=False, ripple_variable_names=None, ripple_as_df=False,
        load_stim_info=False, stim_info_variable_names=None, stim_info_as_df=False,
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
            ret_dict['stim_info'] = sanitize_stim_info(loadtablefrommat(ret_dict['stim_info']), nev_spikes)
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
