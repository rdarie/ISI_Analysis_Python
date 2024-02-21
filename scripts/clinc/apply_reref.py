import pandas as pd
from pathlib import Path
import json
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan, emg_sample_rate, dsi_trig_sample_rate

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
routing_config_info['config_start_time'] = routing_config_info['config_start_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))
routing_config_info['config_end_time'] = routing_config_info['config_end_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))

with open(folder_path / 'analysis_metadata/reref_lookup.json', 'r') as f:
    reref_lookup_dict = json.load(f)
for file_name in routing_config_info['child_file_name']:
    if file_name in reref_lookup_dict:
        reref_lookup = reref_lookup_dict[file_name]
    else:
        continue
    print(f"rereferencing {file_name}")
    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    reref_df = pd.DataFrame(0, index=clinc_df.index, columns=[key for key in reref_lookup.keys()])
    for key, value in reref_lookup.items():
        reref_df[key] = clinc_df[key] - clinc_df[value]
    reref_df.to_parquet(folder_path / (file_name + '_clinc_reref.parquet'), engine='fastparquet')
    print('\tDone')
