import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml

def parse_mb_stim_csv(file_path):
    params_df = pd.read_csv(file_path, header=1, index_col=0)
    params_df.index.name = 'cactus_chan'
    params_df.columns = [
        'amp', 'pw', 'DZ0', 'DELTA', 'DZ1',
        'M', 'DZ2', 'THP_DEL', 'N', 'DZ3', 'P', 'RAMP_SCALE_OPT'
        ]
    null_mask = (params_df == 0).all(axis=1)
    params_df = params_df.loc[~null_mask, :].astype(int)

    params_df['amp'] = params_df['amp'] * 12  # uA
    params_df['pw'] = params_df['pw'] * 10  # usec
    params_df['DZ0'] = params_df['DZ0'] * 10  # usec
    params_df['DELTA'] = params_df['DELTA'] * 10  # usec
    params_df['DZ1'] = params_df['DZ1'] * 80  # usec
    params_df['DZ2'] = params_df['DZ2'] * 40.96  # msec
    params_df['THP_DEL'] = params_df['THP_DEL'] * 40.96  # msec
    params_df['DZ3'] = params_df['DZ3'] * 20.97  # sec

    def parse_one_row(row):
        if row['M'] > 1:
            freq = (row['DZ1'] * 1e-6) ** -1
            train_dur = (row['M'] - 1) * row['DZ1'] * 1e-6
            train_period = row['DZ2'] * 1e-3
        else:
            freq = (row['DZ2'] * 1e-3) ** -1
            train_dur = (row['N'] - 1) * row['DZ2'] * 1e-3
            train_period = row['DZ3']
        return freq, train_dur, train_period

    params_df[['freq', 'train_dur', 'train_period']] = pd.DataFrame(
        params_df.apply(parse_one_row, axis='columns').tolist(),
        index=params_df.index,
        columns=['freq', 'train_dur', 'train_period'])

    return params_df

sid_to_cactus = {
    4: 13,
    9: 14,
    10: 15,
    8: 16,
    13: 5
}
cactus_to_sid = {
    c: s for s, c in sid_to_cactus.items()
}
# clinc_sample_rate = 36931.8
filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
# file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
iti_lookup = {
    "MB_1699558933_985097": 3,
    "MB_1699560317_650555": 0.5,
}

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
iti_lookup = {
    fn: 1. for fn in file_name_list
}
file_name_list = ['MB_1700673350_780580']

for file_name in file_name_list:
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    clinc_sample_rate = (float(np.median(np.diff(clinc_df.index))) * 1e-9) ** -1

    high_pass_filter = False
    if high_pass_filter:
        filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), clinc_sample_rate)
        clinc_df = pd.DataFrame(
            signal.sosfiltfilt(filterCoeffs, clinc_df, axis=0),
            index=clinc_df.index, columns=clinc_df.columns)

    get_derivative = True
    if get_derivative:
        clinc_df = clinc_df.diff().fillna(method='bfill')

    scaler = StandardScaler()
    scaler.fit(clinc_df)
    clinc_df = pd.DataFrame(
        scaler.transform(clinc_df),
        index=clinc_df.index, columns=clinc_df.columns)

    artifact_signal = (clinc_df ** 2).mean(axis='columns').to_frame(name='average_zscore')
    artifact_signal.to_parquet(folder_path / (file_name + '_average_zscore.parquet'))

    signal_thresh = 8.
    temp = pd.Series(artifact_signal['average_zscore'].to_numpy())
    cross_index, cross_mask = getThresholdCrossings(
        temp, thresh=signal_thresh, fs=clinc_sample_rate, iti=iti_lookup[file_name], absVal=False, keep_max=False)
    align_timestamps = artifact_signal.index[cross_mask].copy()

    with open(folder_path / 'stim_csv_lookup.json', 'r') as f:
        stim_info_json = json.load(f)
    with open(folder_path / 'yaml_lookup.json', 'r') as f:
        yml_path = json.load(f)[file_name]
    with open(folder_path / yml_path, 'r') as f:
        routing_info_str = ''.join(f.readlines())
        routing_info = yaml.safe_load(routing_info_str.replace('\t', '  '))

    sid_to_eid = {
        c['sid']: c['eid'] for c in routing_info['data']['contacts']
        }

    for idx, entry in enumerate(stim_info_json[file_name]):
        full_path = folder_path / entry['file']
        params_df = parse_mb_stim_csv(full_path)
        params_df.index = [sid_to_eid[cactus_to_sid[c]] for c in params_df.index]
        params_df.index.name = 'eid'
        stim_info_json[file_name][idx]['params'] = params_df
    def assign_stim_metadata(t, stim_dict_list=None):
        for row in stim_dict_list:
            if (t > row['start_time']) & (t < row['end_time']):
                return row['params']
        return None

    meta_dict = {}
    for ts in align_timestamps:
        ts_sec = ts.total_seconds()
        meta_dict[ts] = assign_stim_metadata(ts_sec, stim_info_json[file_name])

    stim_info = pd.concat(meta_dict, names=['timestamp', 'eid'])
    stim_info.to_parquet(folder_path / (file_name + '_stim_info.parquet'))
