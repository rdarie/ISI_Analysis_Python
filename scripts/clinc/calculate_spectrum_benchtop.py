import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
from pathlib import Path
from isicpy.utils import getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, sid_to_intan, emg_sample_rate, dsi_trig_sample_rate

sns.set(
    context='paper', style='white',
    palette='deep', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={
        'figure.dpi': 300, 'savefig.dpi': 300,
        'lines.linewidth': .5,
        'lines.markersize': 2.,
        'patch.linewidth': .5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        "xtick.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 1,
        "axes.labelsize": 5,
        "axes.titlesize": 6,
        "axes.titlepad": 1,
        "xtick.labelsize": 5,
        'xtick.major.pad': 1,
        'xtick.major.size': 2,
        "ytick.labelsize": 5,
        "ytick.major.pad": 1,
        'ytick.major.size': 2,
        "legend.fontsize": 5,
        "legend.title_fontsize": 6,
        }
    )
file_name = 'MB_1705952197_530018'
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401221300-Benchtop")

clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
valid_data_df = pd.read_parquet(folder_path / (file_name + '_valid_clinc_data.parquet'))

data = clinc_df.loc[:, "S15"].copy()
data.index = (data.index - data.index[0]).total_seconds()
d_data = data.diff()
d_data_abs = d_data.abs()

### Sweep params taken from neurodac git repo
inset_duration = 0.05
min_duration = 3.
pause_duration = 2.
min_num_cycles_per_freq = 100
freqs_to_sweep = np.logspace(np.log10(0.5), np.log10(15e3), 100)
####

durations = [
    min_num_cycles_per_freq * f ** -1 +
    min_duration + 2 * inset_duration
    for f in freqs_to_sweep]

sleep_fudge_factor = 1.75
threshold = 75
t_start = [data.index[data > threshold][0]]
t_end = []
left_sweep, right_sweep = -2, 2

for idx, nominal_dur in enumerate(durations):
    # get end of last one
    nominal_ts = t_start[-1] + nominal_dur
    mask = (data.index >= nominal_ts + left_sweep) & (data.index <= nominal_ts + right_sweep)
    print(f'Looking for train end from {data[mask].index[0]:.2f} to {data[mask].index[-1]:.2f}')
    crossIdx, crossMask = getThresholdCrossings(
        d_data_abs[mask], thresh=threshold, absVal=False,
        edgeType='rising', fs=clinc_sample_rate, iti=None,
        keep_max=True, itiWiggle=0.05,
        plotting=False, plot_opts=dict())
    print(f'Adding {crossIdx[-1]:.2f} to t_end')
    t_end.append(crossIdx[-1])
    if idx < len(durations) - 1:
        # get start of next one
        nominal_ts = t_end[-1] + pause_duration + sleep_fudge_factor
        mask = (data.index >= t_end[-1] + sleep_fudge_factor) & (data.index <= nominal_ts + right_sweep)
        print(f'Looking for train start from {data[mask].index[0]:.2f} to {data[mask].index[-1]:.2f}')
        crossIdx, crossMask = getThresholdCrossings(
            d_data_abs[mask], thresh=threshold, absVal=False,
            edgeType='rising', fs=clinc_sample_rate, iti=None,
            keep_max=True, itiWiggle=0.05,
            plotting=False, plot_opts=dict())
        print(f'Adding {crossIdx[0]:.2f} to t_start')
        t_start.append(crossIdx[0])
    if idx in (10, 25, 50, 75, 98):
        mask = (data.index >= t_end[-1] + left_sweep) & (data.index <= t_start[-1] + right_sweep)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(data[mask])
        ax[0].axvline(t_end[-1], c='r')
        ax[0].axvline(t_start[-1], c='g')
        ax[1].plot(d_data[mask])
        ax[1].axvline(t_end[-1], c='r')
        ax[1].axvline(t_start[-1], c='g')

plot_start = 2000
plot_dur = 500
fig, ax = plt.subplots(2, 1, sharex=True)
plot_mask = (d_data.index >= plot_start) & (d_data.index <= plot_start + plot_dur)
ax[0].plot(d_data.loc[plot_mask])
ax[1].plot(data.loc[plot_mask])
plt.show()

fig, ax = plt.subplots()
start_mask = data.index.isin(t_start)
end_mask = data.index.isin(t_end)
ax.plot(data)
ax.plot(data[start_mask], 'g*')
ax.plot(data[end_mask], 'r*')

epoched_data = {}
rms = {}
rms_std = {}
trim_edges = 0.5

for idx, (t_on, t_off) in enumerate(zip(t_start, t_end)):
    print(f'Epoch from {t_on:.2f} to {t_off:.2f}')
    '''
    if idx == 6:
        # 6th sample has a big amplifier pop towards the end, which distorts the RMS
        mask = (data.index >= t_on + trim_edges) & (data.index <= t_off - trim_edges - 2)
    else:
        mask = (data.index >= t_on + trim_edges) & (data.index <= t_off - trim_edges)
        '''
    mask = (data.index >= t_on + trim_edges) & (data.index <= t_off - trim_edges)
    epoched_data[t_on] = data.loc[mask].copy()
    rms[t_on] = np.sqrt((epoched_data[t_on] ** 2).median())
    rms_std[t_on] = np.sqrt((epoched_data[t_on] ** 2).std())

rms_df = pd.Series(rms).to_frame(name='rms')
rms_df['normalized_rms'] = 10 * np.log10(rms_df['rms'] / rms_df['rms'].max())
rms_df['freq'] = freqs_to_sweep
rms_df['std'] = rms_std

pdf_path = folder_path / ('clinc_spectrum.pdf')
with PdfPages(pdf_path) as pdf:
    fig, ax = plt.subplots()
    ax.plot(rms_df['freq'], rms_df['normalized_rms'])
    ax.set_xlabel('Input Frequency (Hz.)')
    ax.set_ylabel('Output RMS (dB.)')
    ax.set_xscale('log')
    fig.align_labels()
    pdf.savefig(bbox_inches='tight', pad_inches=0)
    plt.show()


pdf_path = folder_path / ('clinc_spectrum_w_examples.pdf')
png_path = folder_path / ('clinc_spectrum_w_examples.png')
with PdfPages(pdf_path) as pdf:
    show_indices = [2, 25, 50, 75, 98]
    # all_colors = sns.color_palette(palette='deep', n_colors=8)
    # color_list = [all_colors[i] for i in [0, 1, 2, 3, 6]]
    color_list = sns.cubehelix_palette(start=.5, rot=-.5, n_colors=len(show_indices))
    fig = plt.figure(figsize=(1.8, 4), layout='constrained')
    '''
    fig, ax = plt.subplots(
        len(show_indices) + 1, 1,
        figsize=(3, 6), height_ratios=[4] + len(show_indices) * [1],
        gridspec_kw=dict(
            hspace=.45
        ))
        '''
    gs0 = mpl.gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 1.75],
        )
    gs00 = gs0[0].subgridspec(1, 1)
    gs01 = gs0[1].subgridspec(len(show_indices), 1)
    ax = [fig.add_subplot(gs00[0, 0])]
    ax[0].plot(rms_df['freq'], rms_df['normalized_rms'])
    ax[0].scatter(
        rms_df['freq'].iloc[show_indices], rms_df['normalized_rms'].iloc[show_indices],
        c=color_list, s=9, zorder=2)
    ax[0].set_xlabel('Input Frequency (Hz.)')
    ax[0].set_ylabel('Recorded Signal RMS (dB.)')
    ax[0].set_xscale('log')
    ax[0].grid(visible=True)
    # ax[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x: >7.2g}'))
    # longest_period = freqs_to_sweep[show_indices[0]] ** -1
    # bounds_adjust = [
    #     (0.4, 0.4), (0.133, 0.273), (10.2e-3, 10.2e-3), (3.7e-4, 10.2e-4), (6.4e-5, 7.5e-5)]  # adjust the time bound slightly, to align x ticks
    for ax_idx, t_idx in enumerate(show_indices):
        this_ax = fig.add_subplot(gs01[ax_idx, 0])
        ax.append(this_ax)
        t = rms_df.index[t_idx]
        period = freqs_to_sweep[t_idx] ** -1
        print(f"period = {period:.6f}")
        # adjust_start, adjust_stop = bounds_adjust[ax_idx]
        adjust_start, adjust_stop = 0, 0
        mask = (
                (epoched_data[t].index >= epoched_data[t].index[0] + adjust_start) &
                (epoched_data[t].index < epoched_data[t].index[0] + 10 * period + adjust_stop)
        )
        y = epoched_data[t][mask].copy()
        y.index -= y.index[0]
        this_ax.plot(y, c=color_list[ax_idx])
        this_ax.set_ylim(-800, 800)
        this_ax.set_xlim(y.index[0], y.index[-1])
        this_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[2, 5], min_n_ticks=3))
        this_ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2g}'))
        # this_ax.grid(visible=True)
        this_ax.text(1, 1, f"{freqs_to_sweep[t_idx]:.1f} Hz", transform=this_ax.transAxes, va='bottom', ha='right', fontsize=5)
    ax[3].set_ylabel('Recorded Signal (uV)')
    ax[-1].set_xlabel('Time (sec.)')
    sns.despine(fig)
    # fig.align_labels()
    pdf.savefig(pad_inches=25e-3)
    fig.savefig(png_path, pad_inches=25e-3)
    plt.show()





plt.plot(epoched_data[rms_df.index[6]])