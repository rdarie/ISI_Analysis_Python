import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import pandas as pd
import numpy as np
from pathlib import Path
import json
from isicpy.lookup_tables import HD64_topo, HD64_labels, eid_palette, eids_ordered_xy
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import os

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={"xtick.bottom": True}
    )

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
routing_config_info = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
routing_config_info['config_start_time'] = routing_config_info['config_start_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))
routing_config_info['config_end_time'] = routing_config_info['config_end_time'].apply(
    lambda x: pd.Timestamp(x, tz='GMT'))

with open(folder_path / 'analysis_metadata/reref_lookup.json', 'r') as f:
    reref_lookup_dict = json.load(f)

# all units in mm
py = 5.  # y pitch
px = 1.5  # x pitch
wc = 1.  # contact width
lc = 3.6  # contact length
w_elec = 13.5
l_elec = 40.5
l_bottom = 15
x_offset = 1
y_offset = 1

x = x_offset + np.arange(0, HD64_topo.shape[1]) * px
y = y_offset + np.arange(0, HD64_topo.shape[0])[::-1] * py
xv, yv = np.meshgrid(x, y, indexing='xy')
xv = pd.DataFrame(xv, index=HD64_topo.index, columns=HD64_topo.columns)
yv = pd.DataFrame(yv, index=HD64_topo.index, columns=HD64_topo.columns)

master_linewidth = 1

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

for yml_path, this_routing in routing_config_info.groupby('yml_path'):
    if yml_path == 'nan':
        continue

    reref_lookup = None
    for child_name in this_routing['child_file_name']:
        if child_name in reref_lookup_dict:
            reref_lookup = reref_lookup_dict[child_name]
            break
    if reref_lookup is None:
        continue

    cfg_name = Path(yml_path).stem
    print(f'{cfg_name} used by {this_routing["child_file_name"].to_list()}')
    pdf_path = folder_path / "figures" / (f'{cfg_name}_reref_hd64_map.pdf')
    active_eid_labels = this_routing['clinc_col_names'].iloc[0]


    color_lookup = {}
    text_lookup = {}


    for eid_plus, eid_minus in reref_lookup.items():
        color_lookup[eid_plus] = eid_palette[eid_plus]
        color_lookup[eid_minus] = eid_palette[eid_plus]
        text_lookup[eid_plus] = '+'
        text_lookup[eid_minus] = '-'

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(2, 7))
        patch_artist = mpatches.Arc(
            (w_elec / 2, l_elec), w_elec, w_elec,
            theta1=0, theta2=180,
            ec='k', fc='none', lw=master_linewidth
        )
        ax.add_artist(patch_artist)
        ax.plot([0, 0], [-l_bottom, l_elec], lw=master_linewidth, c='k')
        ax.plot([w_elec, w_elec], [-l_bottom, l_elec], lw=master_linewidth, c='k')
        ax.plot([0, w_elec], [-l_bottom, -l_bottom], lw=master_linewidth, c='k')

        for row in HD64_topo.index:
            for col in HD64_topo.columns:
                eid = HD64_topo.loc[row, col]
                eid_label = HD64_labels.loc[row, col]
                if eid > -1:
                    if eid_label in color_lookup:
                        patch_artist = mpatches.FancyBboxPatch(
                            (xv.loc[row, col], yv.loc[row, col]), wc, lc,
                            lw=master_linewidth, ec="k", fc=color_lookup[eid_label],
                            boxstyle=mpatches.BoxStyle(
                                "Round", pad=0, rounding_size=.5)
                            )
                        ax.add_artist(patch_artist)
                        ax.text(
                            xv.loc[row, col] + wc / 2, yv.loc[row, col] + lc / 2,
                            text_lookup[eid_label], size=8, fontweight='bold',
                            # transform=ax.transAxes, size="large", color="k",
                            horizontalalignment="center", verticalalignment="center")
                    else:
                        patch_artist = mpatches.FancyBboxPatch(
                            (xv.loc[row, col], yv.loc[row, col]), wc, lc,
                            lw=master_linewidth, ec='k', fc="none",
                            boxstyle=mpatches.BoxStyle(
                                "Round", pad=0, rounding_size=.5)
                            )
                        ax.add_artist(patch_artist)

        ax.set_xlim(-0.5, w_elec + 0.5)
        ax.set_ylim(-0.5 - l_bottom, l_elec + w_elec + 0.5)

        ax.set_axis_off()
        pdf.savefig()
        plt.show()