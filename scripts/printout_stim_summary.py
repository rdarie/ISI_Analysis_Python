import os
from isicpy.utils import load_synced_mat, closestSeries
from isicpy.lookup_tables import emg_montages
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pdb
import traceback
from tqdm import tqdm
import cloudpickle as pickle
from sklearn.preprocessing import StandardScaler

this_emg_montage = emg_montages['lower']
# folder_name = "Day12_PM"
# blocks_list = [3, 4]
folder_name = "Day11_PM"
blocks_list = [2, 3]
# folder_name = "Day8_AM"
# blocks_list = [1, 2, 3, 4]

data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{folder_name}")
html_folder = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/5_Figures/{folder_name}")
html_path = html_folder / "stim_summary.html"

this_emg_montage = emg_montages['lower_v2']
all_stim_info = {}
for block_idx in tqdm(blocks_list):
    file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
    data_dict = load_synced_mat(
        file_path, load_stim_info=True,
        )
    if data_dict['stim_info'] is not None:
        all_stim_info[block_idx] = data_dict['stim_info']

stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')

stim_info_df.loc[:, 'freq_binned'] = pd.cut(stim_info_df['freq'], bins=10)
stim_info_df.loc[:, 'amp_binned'] = pd.cut(stim_info_df['amp'], bins=10)
##  ['block', 'elecConfig_str', 'amp_binned', 'freq_binned']

# stim_info_df.groupby(['block', 'elecConfig_str']).quantile(q=[0.1, 0.9])['freq']
counts_df = stim_info_df.reset_index().groupby(['block', 'elecConfig_str']).count().iloc[:, 0]
counts_df = counts_df.loc[counts_df != 0]
counts_df.name = 'count'
counts_df = counts_df.reset_index()
counts_df.sort_values(by=['block', 'count'])

cssTableStyles = [
    {
        'selector': 'th',
        'props': [
            ('border-style', 'solid'),
            ('border-color', 'black')]},
    {
        'selector': 'td',
        'props': [
            ('border-style', 'solid'),
            ('border-color', 'black')]},
    {
        'selector': 'table',
        'props': [
            ('border-collapse', 'collapse')
        ]}
]
cm = sns.dark_palette("green", as_cmap=True)

dfStyler = (
    counts_df
    .style
    .background_gradient(cmap=cm)
    .set_precision(1)
    .set_table_styles(cssTableStyles)
)

with open(html_path, 'w') as _file:
    _file.write(dfStyler.render())
