import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_utils import parse_mb_stim_csv, assign_stim_metadata
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_cactus, cactus_to_sid
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import yaml

filterOpts = {
    'high': {
        'Wn': 1000.,
        'N': 2,
        'btype': 'high',
        'ftype': 'butter'
    },
}
per_pulse = False

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/202311091300-Phoenix")
file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555"]
iti_lookup = {
    "MB_1699558933_985097": 3,
    "MB_1699560317_650555": 0.5,
}

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name_list = [
    'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
    'MB_1700672668_26337', 'MB_1700673350_780580'
    ]
iti_lookup = {
    fn: 1. for fn in file_name_list
}

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = [
    "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
    "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
]
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]

iti_lookup = {
    fn: .5 for fn in file_name_list
}

for file_name in file_name_list:
    print(f"Processing {file_name}...")
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))

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
    artifact_signal.to_parquet(folder_path / (file_name + '_average_zscore.parquet'), engine='fastparquet')

    t_zero = artifact_signal.index[0]
    artifact_signal.index -= t_zero

    signal_thresh = 8.
    temp = pd.Series(artifact_signal['average_zscore'].to_numpy())

    if per_pulse:
        this_iti = 2e-3
    else:
        this_iti = iti_lookup[file_name]

    cross_index, cross_mask = getThresholdCrossings(
        temp, thresh=signal_thresh, fs=clinc_sample_rate, iti=this_iti, absVal=False, keep_max=False)
    align_timestamps = artifact_signal.index[cross_mask].copy()

    stim_info_csv = pd.read_csv(folder_path / f'{file_name}_log.csv')
    mask = stim_info_csv['CODE'].isin(['stim_load_config', 'usb_stream_stop'])
    relevant_codes = stim_info_csv.loc[mask, :]

    stim_info_list = []
    for idx in range(mask.sum()):
        if relevant_codes.iloc[idx, :]['CODE'] == 'stim_load_config':
            this_stim_config = Path(relevant_codes.iloc[idx, :]['FILENAME']).name
            this_entry = {
                'file': f'./config_files/{this_stim_config}',
                'start_time': relevant_codes.iloc[idx, :]['SEC_ELAPSED'],
                'end_time': relevant_codes.iloc[idx + 1, :]['SEC_ELAPSED']
            }
            stim_info_list.append(this_entry)

    with open(folder_path / 'yaml_lookup.json', 'r') as f:
        yml_path = json.load(f)[file_name]
    with open(folder_path / yml_path, 'r') as f:
        routing_info_str = ''.join(f.readlines())
        routing_info = yaml.safe_load(routing_info_str.replace('\t', '  '))

    sid_to_eid = {
        c['sid']: c['eid'] for c in routing_info['data']['contacts']
        }

    for idx, entry in enumerate(stim_info_list):
        full_path = folder_path / entry['file']
        params_df = parse_mb_stim_csv(full_path)
        params_df.index = [sid_to_eid[cactus_to_sid[c]] for c in params_df.index]
        params_df.index.name = 'eid'
        stim_info_list[idx]['params'] = params_df

    meta_dict = {}
    for ts in align_timestamps:
        ts_sec = ts.total_seconds()
        meta_dict[ts] = assign_stim_metadata(ts_sec, stim_info_list)

    stim_info = pd.concat(meta_dict, names=['timestamp', 'eid'])

    pulse_diffs = (
        pd.Series(
            stim_info.index.get_level_values('timestamp'),
            index=stim_info.index).apply(lambda x: x.total_seconds())
        .diff().fillna(np.inf)
        )

    fudge_factor = 0.9
    is_first = (pulse_diffs > stim_info['train_period'] * fudge_factor)
    train_idx = is_first.cumsum()
    train_rank = pd.Series(0, index=stim_info.index)
    for idx in train_idx.unique():
        this_mask = train_idx == idx
        train_rank.loc[this_mask] = np.arange(this_mask.sum())

    stim_info['train_idx'] = train_idx
    stim_info['rank_in_train'] = train_rank

    stim_info_index = stim_info.index.to_frame().reset_index(drop=True)
    stim_info_index['timestamp'] += t_zero
    stim_info.index = pd.MultiIndex.from_frame(stim_info_index)

    if per_pulse:
        output_filename = file_name + '_stim_info_per_pulse.parquet'
    else:
        output_filename = file_name + '_stim_info.parquet'
    stim_info.to_parquet(folder_path / output_filename, engine='fastparquet')
