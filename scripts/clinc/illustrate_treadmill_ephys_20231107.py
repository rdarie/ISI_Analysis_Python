import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings, mapToDF
from isicpy.lookup_tables import eid_remix_lookup, eids_ordered_xy, eid_palette
from isicpy.clinc_lookup_tables import clinc_paper_matplotlib_rc
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

sns.set(
    context='paper', style='white',
    palette='deep', font='sans-serif',
    font_scale=1, color_codes=True,
    rc=clinc_paper_matplotlib_rc
    )
from matplotlib import pyplot as plt

clinc_sample_interval = pd.Timedelta(27077, unit='ns').to_timedelta64()
clinc_sample_interval_sec = float(clinc_sample_interval) * 1e-9
#  = clinc_sample_interval_sec ** -1
clinc_sample_rate = 36931.8
filterOpts = {
    'low': {
        'Wn': 500.,
        'N': 8,
        'btype': 'low',
        'ftype': 'butter'
    },
}
filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), clinc_sample_rate)

downsample_factor = 10
downsampled_interval = clinc_sample_interval_sec * downsample_factor
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
dsi_block_list = []
file_name = "MB_1699384177"

lfp_path = (file_name + '_clinc.parquet')
pdf_path = folder_path / "figures" / ('treadmill_illustration.pdf')
png_path = folder_path / 'figures' / ('treadmill_illustration.png')

lfp_df = pd.read_parquet(folder_path / lfp_path)
lfp_df.index -= lfp_df.index[0]
lfp_df.index = lfp_df.index.total_seconds()
lfp_df = pd.DataFrame(
    signal.sosfiltfilt(filterCoeffs, lfp_df, axis=0),
    index=lfp_df.index, columns=lfp_df.columns).iloc[::downsample_factor, :]

this_eid_order = [cn for cn in eids_ordered_xy if cn in lfp_df.columns]
this_eid_palette = {lbl: eid_col for lbl, eid_col in eid_palette.items() if lbl in lfp_df.columns}

window_len = 5  # sec
window_len_samples = int(window_len / downsampled_interval)

align_timestamps = {
    1: 'off',
    34: 'on'
}

epoched_dict = {}
t = None
for ts, treadmill in align_timestamps.items():
    this_mask = lfp_df.index > ts
    first_index = np.flatnonzero(this_mask)[0]
    epoched_dict[(ts, treadmill)] = lfp_df.iloc[first_index:first_index + window_len_samples, :].copy()
    if t is None:
        t = np.arange(epoched_dict[(ts, treadmill)].shape[0]) * downsampled_interval
    epoched_dict[(ts, treadmill)].index = t
    epoched_dict[(ts, treadmill)].index.name = 't'

epoched_df = pd.concat(epoched_dict, names=['timestamp', 'treadmill'])
epoched_df = epoched_df - epoched_df.mean()

average_std = epoched_df.std().mean()
dy = 0
for cn in epoched_df.columns:
    epoched_df[cn] = epoched_df[cn] + dy
    dy += 8 * average_std

plot_df = epoched_df.stack().reset_index().rename(columns={0: 'value'})
plot_df['value'] = plot_df['value'] / 1e3
if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")

## substitute renumbered EIDs
this_eid_palette_remix = {eid_remix_lookup[old_name]: c for old_name, c in this_eid_palette.items()}
this_eid_order_remix = [eid_remix_lookup[old_name] for old_name in this_eid_order]
plot_df['eid'] = plot_df.apply(lambda x: eid_remix_lookup[x['eid']], axis='columns')
#####

with PdfPages(pdf_path) as pdf:
    g = sns.relplot(
        data=plot_df,
        row='treadmill',
        x='t', y='value',
        hue='eid', hue_order=this_eid_order_remix, palette=this_eid_palette_remix,
        kind='line', estimator=None,
        facet_kws=dict(legend_out=False, xlim=(0, window_len))
        )
    desired_figsize = (1.95, 3.6)
    g.set_ylabels('Spinal Potential (mV)')
    g.set_xlabels('Time (sec.)')
    g.legend.set_title('Spinal\nChannel')
    g.set_titles(row_template="Treadmill {row_name}")
    g.figure.set_size_inches(desired_figsize)
    sns.move_legend(
        g, 'center right', bbox_to_anchor=(1, 0.5),
        ncols=1)
    for legend_handle in g.legend.legendHandles:
        if isinstance(legend_handle, mpl.lines.Line2D):
            legend_handle.set_lw(4 * legend_handle.get_lw())
    '''
    g.legend._set_loc(7)
    g.legend.set_bbox_to_anchor(
        (1., 0.5), transform=g.figure.transFigure)
    '''
    g.figure.draw_without_rendering()
    legend_approx_width = g.legend.legendPatch.get_width() / g.figure.get_dpi()  # inches
    new_right_margin = 1 - legend_approx_width / desired_figsize[0]
    g.figure.subplots_adjust(right=new_right_margin)
    g.tight_layout(
        pad=25e-2, rect=[0, 0, new_right_margin, 1])
    g.figure.align_labels()
    pdf.savefig()
    g.figure.savefig(png_path)
    plt.show()
    # plt.close()
