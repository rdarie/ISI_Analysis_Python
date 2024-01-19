import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import pandas as pd
import numpy as np
from pathlib import Path
import json
from isicpy.lookup_tables import HD64_topo_list
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={"xtick.bottom": True}
    )

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name_list = ["MB_1702049441_627410", "MB_1702049896_129326"]
file_name = "MB_1702049441_627410"
with open(folder_path / 'reref_lookup.json', 'r') as f:
    reref_lookup = json.load(f)[file_name]

full_palette = sns.color_palette('Paired')
color_lookup = {}
text_lookup = {}
col_order = [
    "E47", "E0", 'E58', "E16", "E59", "E37"]
for c_idx, key in enumerate(col_order):
    value = reref_lookup[key]
    color_lookup[key] = full_palette[c_idx]
    color_lookup[value] = full_palette[c_idx]
    text_lookup[key] = '+'
    text_lookup[value] = '-'

HD64_topo = pd.DataFrame(HD64_topo_list)
HD64_topo.index.name = 'y'
HD64_topo.columns.name = 'x'
HD64_labels = HD64_topo.applymap(lambda x: f"E{x:d}" if (x >= 0) else "")

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

pdf_path = folder_path / "figures" / ('reref_hd64_map.pdf')
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots(figsize=(2, 7))
    patch_artist = mpatches.Arc(
        (w_elec / 2, l_elec), w_elec, w_elec,
        theta1=0, theta2=180,
        ec='k', fc='none', lw=2
    )
    ax.add_artist(patch_artist)
    ax.plot([0, 0], [-l_bottom, l_elec], c='k')
    ax.plot([w_elec, w_elec], [-l_bottom, l_elec], c='k')
    ax.plot([0, w_elec], [-l_bottom, -l_bottom], c='k')

    for row in HD64_topo.index:
        for col in HD64_topo.columns:
            eid = HD64_topo.loc[row, col]
            eid_label = HD64_labels.loc[row, col]
            if eid > -1:
                if eid_label in color_lookup:
                    patch_artist = mpatches.FancyBboxPatch(
                        (xv.loc[row, col], yv.loc[row, col]), wc, lc,
                        ec="k", fc=color_lookup[eid_label],
                        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=.5)
                        )
                    ax.add_artist(patch_artist)
                    ax.text(
                        xv.loc[row, col] + wc / 2, yv.loc[row, col] + lc / 2,
                        text_lookup[eid_label],
                        # transform=ax.transAxes, size="large", color="k",
                        horizontalalignment="center", verticalalignment="center")
                else:
                    patch_artist = mpatches.FancyBboxPatch(
                        (xv.loc[row, col], yv.loc[row, col]), wc, lc, ec='k', fc="none",
                        boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=.5)
                        )
                    ax.add_artist(patch_artist)

    ax.set_xlim(-0.5, w_elec + 0.5)
    ax.set_ylim(-0.5 - l_bottom, l_elec + w_elec + 0.5)

    ax.set_axis_off()
    pdf.savefig()
    plt.show()