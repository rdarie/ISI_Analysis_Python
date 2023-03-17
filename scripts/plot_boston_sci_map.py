from isicpy.third_party.pymatreader import read_mat
from isicpy.utils import load_synced_mat
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
import pandas as pd
import numpy as np
from itertools import product
base_path = Path("/gpfs/home/rdarie/isi_analysis/ISI_Code_On_The_Box/Analysis/Visualizers")

map_mat_path = base_path / "dual_array_map.mat"
dual_array_px_map = read_mat(map_mat_path, variable_names=['dual_array_px_map'])['dual_array_px_map']
pil_im = Image.open(base_path / "DualArrays.png", 'r')

def plot_stim_map(which_day, blocks_list, min_count=10):
    data_path = Path(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/3_Preprocessed_Data/{which_day}")
    blocks_list_str = ', '.join(f"{block_idx}" for block_idx in blocks_list)
    all_stim_info = {}
    for block_idx in blocks_list:
        file_path = data_path / f"Block{block_idx:0>4d}_Synced_Session_Data.mat"
        data_dict = load_synced_mat(
            file_path,
            load_stim_info=True, split_trains=True,
        )
        all_stim_info[block_idx] = data_dict['stim_info']

    stim_info_df = pd.concat(all_stim_info, names=['block', 'timestamp_usec'])
    stim_info_df.loc[:, 'elecConfig_str'] = stim_info_df.apply(lambda x: f'-{x["elecCath"]}+{x["elecAno"]}', axis='columns')
    counts_df = stim_info_df.reset_index().groupby(['block', 'elecConfig_str', 'elecCath', 'elecAno']).count().iloc[:, 0]
    counts_df = counts_df.loc[counts_df != 0]
    counts_df.name = 'count'
    counts_df = counts_df.loc[counts_df > min_count]
    counts_df = counts_df.reset_index()
    counts_df.sort_values(by=['block', 'count'])

    fig, ax = plt.subplots()
    ax.imshow(np.asarray(pil_im) * 0.4 / 255)

    for row_idx, stim_combo in counts_df.iterrows():
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for c, a in product(stim_combo['elecCath'], stim_combo['elecAno']):
            x0 = np.mean(dual_array_px_map['x'][a-1])
            y0 = np.mean(dual_array_px_map['y'][a-1])
            x1 = np.mean(dual_array_px_map['x'][c-1])
            y1 = np.mean(dual_array_px_map['y'][c-1])
            x_min = min(x_min, x0, x1)
            y_min = min(y_min, y0, y1)
            x_max = max(x_max, x0, x1)
            y_max = max(y_max, y0, y1)
            ax.annotate(
                "",
                xy=(x0, y0), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->"))
        ax.text(
            (x_min + x_max) / 2, (y_min + y_max) / 2,
            stim_combo['elecConfig_str'],
            ha='left', va='bottom'
            )
        fig.suptitle(f"{which_day}: blocks {blocks_list_str}")
    return


plot_stim_map("Day8_AM", [4])
plot_stim_map("Day11_PM", [2, 3])
plot_stim_map("Day12_PM", [4])

plt.show()
