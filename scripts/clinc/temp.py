import json
import pandas as pd
from pathlib import Path

example = {
    'MB_1699558933_985097': [{
    'start_time': 0,
    'end_time': 1,
    'amp': 0,
    'freq': 100,
    'pw': 0,
}]}

file_path = '/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix/stim_info.json'

with open(file_path, 'w') as f:
    json.dump(example, f, indent=4)

folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
file_name = "MB_1699558933_985097_f.mat"
clinc_df = pd.read_parquet(folder_path / file_name.replace('.mat', '_clinc.parquet'))

filtered = pd.DataFrame(
    signal.sosfiltfilt(filterCoeffsClinc, clinc_df, axis=0),
    index=clinc_df.index, columns=clinc_df.columns)

plt.plot(clinc_df.iloc[:int(4e4), 0])
plt.plot(filtered.iloc[:int(4e4), 0])

