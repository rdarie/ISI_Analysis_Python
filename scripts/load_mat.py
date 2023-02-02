import pandas as pd
import numpy as np
from pathlib import Path
from pymatreader.utils import _hdf5todict as hdf5todict
import h5py

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
            if vicon_as_df:
                if 'Points' in ret_dict['vicon']:
                    x = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][0, :, :].T,
                        index=ret_dict['vicon']['Points']['Time'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    y = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][1, :, :].T,
                        index=ret_dict['vicon']['Points']['Time'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    z = pd.DataFrame(
                        ret_dict['vicon']['Points']['Data'][2, :, :].T,
                        index=ret_dict['vicon']['Points']['Time'],
                        columns=ret_dict['vicon']['Points']['Labels'])
                    df = pd.concat({'x': x, 'y': y, 'z': z}, axis='columns')
                    df.columns.names = ['axis', 'label']
                    df.index.name = 'time'
                    df = df.swaplevel(axis='columns')
                    df.sort_index(axis='columns', inplace=True, level=['label', 'axis'])
                    ret_dict['vicon']['Points'] = df
                if 'Devices' in ret_dict['vicon']:
                    df = pd.DataFrame(
                        ret_dict['vicon']['Devices']['Data'],
                        index=ret_dict['vicon']['Devices']['Time']
                    )
                    df.index.name = 'time'
                    df.columns.name = 'label'
                    ret_dict['vicon']['Devices'] = df
                if 'EMG' in ret_dict['vicon']:
                    df = pd.DataFrame(
                        ret_dict['vicon']['EMG']['Data'],
                        index=ret_dict['vicon']['EMG']['Time']
                    )
                    df.index.name = 'time'
                    df.columns.name = 'label'
                    ret_dict['vicon']['EMG'] = df
        if load_ripple:
            ret_dict['ripple'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Ripple_Data'],
                ignore_fields=ignore_fields, variable_names=ripple_variable_names)
            if 'NEV' in ret_dict['ripple']:
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
                # sanitize electrode metadata
                for key, list_of_info in ret_dict['ripple']['NS5']['ElectrodesInfo'].items():
                    if key in ['Label', 'AnalogUnits']:
                        ret_dict['ripple']['NS5']['ElectrodesInfo'][key] = [
                            None if isinstance(lbl, np.ndarray) else lbl.strip().replace('\x00', '')
                            for lbl in list_of_info
                        ]
            if ripple_as_df:
                if 'NF7' in ret_dict['ripple']:
                    n_rows = ret_dict['ripple']['NF7']['time'].shape[0]
                    n_cols = len(ret_dict['ripple']['NF7']['AnalogWaveforms'])
                    df = pd.DataFrame(
                        np.zeros((n_rows, n_cols,)),
                        index=ret_dict['ripple']['NF7']['time'],
                        columns=ret_dict['ripple']['NF7']['AnalogEntityLabels'])
                    for idx, data_col in enumerate(ret_dict['ripple']['NF7']['AnalogWaveforms']):
                        df.iloc[:, idx] = data_col
                    df.index.name = 'time'
                    df.columns.name = 'electrode'
                    ret_dict['ripple']['NF7'] = df
                if 'NS5' in ret_dict['ripple']:
                    column_names = ret_dict['ripple']['NS5']['ElectrodesInfo']['Label']
                    electrodes_info = pd.DataFrame(ret_dict['ripple']['NS5']['ElectrodesInfo'])
                    ret_dict['ripple']['NS5_ElectrodesInfo'] = electrodes_info
                    meta_tags = ret_dict['ripple']['NS5']['MetaTags'].copy()
                    ret_dict['ripple']['NS5_MetaTags'] = meta_tags
                    df = pd.DataFrame(
                        ret_dict['ripple']['NS5']['Data'].T,
                        index=ret_dict['ripple']['NS5']['time'],
                        columns=column_names)
                    df.index.name = 'time'
                    df.columns.name = 'electrode'
                    ret_dict['ripple']['NS5'] = df
                if 'NEV' in ret_dict['ripple']:
                    meta_tags = ret_dict['ripple']['NEV']['MetaTags'].copy()
                    ret_dict['ripple']['NEV_MetaTags'] = meta_tags
                    electrodes_info = pd.DataFrame(ret_dict['ripple']['NEV']['ElectrodesInfo'])
                    ret_dict['ripple']['NEV_ElectrodesInfo'] = electrodes_info
                    waveform_unit = ret_dict['ripple']['NEV']['Data']['Spikes'].pop('WaveformUnit')
                    waveform_df = pd.DataFrame(ret_dict['ripple']['NEV']['Data']['Spikes'].pop('Waveform'))
                    df = pd.DataFrame(ret_dict['ripple']['NEV']['Data']['Spikes'])
        if load_stim_info:
            ret_dict['stim_info'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['StimInfo'],
                ignore_fields=ignore_fields, variable_names=stim_info_variable_names)
        if load_all_logs:
            ret_dict['all_logs'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['AllLogs'],
                ignore_fields=ignore_fields, variable_names=all_logs_variable_names)
        if load_meta:
            ret_dict['meta'] = hdf5todict(
                hdf5_file['Synced_Session_Data']['Meta'],
                ignore_fields=ignore_fields, variable_names=meta_variable_names)
    return ret_dict


data_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/Day11_PM")
file_path = data_path / "Block0006_Synced_Session_Data.mat"
data_dict = load_synced_mat(
    file_path,
    load_vicon=True, vicon_as_df=True,
    load_ripple=True, ripple_variable_names=['NEV'], ripple_as_df=True)