import matplotlib as mpl
# mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import json
from isicpy.utils import makeFilterCoeffsSOS, getThresholdCrossings
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from scipy import signal
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scaleogram as scg
import pywt
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

from matplotlib import pyplot as plt

lfp_dict = {}
folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
file_name = "MB_1699383052_618936"
lfp_path = file_name + '_clinc.parquet'
lfp_dict['no_treadmill'] = pd.read_parquet(folder_path / lfp_path)
lfp_dict['no_treadmill'].index = np.arange(lfp_dict['no_treadmill'].shape[0]) / clinc_sample_rate
lfp_dict['no_treadmill'] = lfp_dict['no_treadmill'].loc[lfp_dict['no_treadmill'].index < 3, :]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
file_name = 'MB_1700670158_174163'
lfp_path = file_name + '_clinc.parquet'
lfp_dict['treadmill_in_room'] = pd.read_parquet(folder_path / lfp_path)
lfp_dict['treadmill_in_room'].index = np.arange(lfp_dict['treadmill_in_room'].shape[0]) / clinc_sample_rate
lfp_dict['treadmill_in_room'] = lfp_dict['treadmill_in_room'].loc[lfp_dict['treadmill_in_room'].index < 3, :]

folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
file_name = 'MB_1702051241_224335'
lfp_path = file_name + '_clinc.parquet'
lfp_dict['on_treadmill'] = pd.read_parquet(folder_path / lfp_path)
lfp_dict['on_treadmill'].index = np.arange(lfp_dict['on_treadmill'].shape[0]) / clinc_sample_rate
mask = (lfp_dict['on_treadmill'].index > 30) & (lfp_dict['on_treadmill'].index < 33)
lfp_dict['on_treadmill'] = lfp_dict['on_treadmill'].loc[mask, :]
lfp_dict['on_treadmill'].index = lfp_dict['on_treadmill'].index - lfp_dict['on_treadmill'].index[0]

wavelet = 'cmor2-10'
freqs = np.linspace(19000, 100, 300)
scales = pywt.frequency2scale(wavelet, freqs / clinc_sample_rate, precision=12)

pdf_path = folder_path / "treadmill_noise_profiles.pdf"
with PdfPages(pdf_path) as pdf:
    chan = 'E45'
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    y = lfp_dict['no_treadmill'][chan].to_numpy()
    t = lfp_dict['no_treadmill'].index.to_numpy()
    cwt = scg.CWT(
        t, signal=y, scales=scales, wavelet=wavelet,
        cwt_fun_args=dict(precision=12, method='fft'))
    scg.cws(
        cwt, ax=ax[0], title='no_treadmill', cscale='log', cbar=False,  # figsize=(12, 6),
        cmap='viridis', yscale='linear', coi=False, clim=(1, 200),
        ylabel="Frequency (Hz)", xlabel='Time (s)', yaxis='frequency')

    y = lfp_dict['treadmill_in_room'][chan].to_numpy()
    t = lfp_dict['treadmill_in_room'].index.to_numpy()
    cwt = scg.CWT(
        t, signal=y, scales=scales, wavelet=wavelet,
        cwt_fun_args=dict(precision=12))
    scg.cws(
        cwt, ax=ax[1], title='treadmill_in_room', cscale='log', cbar=False,  # figsize=(12, 6),
        cmap='viridis', yscale='linear', coi=False, clim=(1, 200),
        ylabel="", xlabel='Time (s)', yaxis='frequency')
    ax[1].set_yticklabels([])

    y = lfp_dict['on_treadmill'][chan].to_numpy()
    t = lfp_dict['on_treadmill'].index.to_numpy()
    cwt = scg.CWT(
        t, signal=y, scales=scales, wavelet=wavelet,
        cwt_fun_args=dict(precision=12))
    scg.cws(
        cwt, ax=ax[2], title='on_treadmill', cscale='log',  # figsize=(12, 6),
        cmap='viridis', yscale='linear', coi=False, clim=(1, 200),
        ylabel="", xlabel='Time (s)', yaxis='frequency')
    ax[2].set_yticklabels([])
    pdf.savefig()
    plt.close()
