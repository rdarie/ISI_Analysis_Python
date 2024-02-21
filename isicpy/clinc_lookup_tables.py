emg_sample_rate = 500
dsi_trig_sample_rate = 1000
clinc_sample_rate = 36931.8
sid_to_intan = {
    1: 8,
    22: 12,
    18: 9,
    19: 7,
    23: 4,
    16: 6,
    15: 10,
    12: 27,
    11: 17,
    6: 21,
    14: 25,
    7: 5,
    0: 13
}

sid_to_label = {
    1: 'S1_S3',
    22: 'S22',
    18: 'S18',
    19: 'S19',
    23: 'S23',
    16: 'S16',
    15: 'S15',
    12: 'S12_S20',
    11: 'S11',
    6: 'S6',
    14: 'S14',
    7: 'S7',
    0: 'S0_S2'
}

sid_to_cactus = {
    4: 13,
    9: 14,
    10: 15,
    8: 16,
    13: 5
}
cactus_to_sid = {
    c: s for s, c in sid_to_cactus.items()
}

dsi_mb_clock_offsets = {
    '202312080900-Phoenix': -6,
}

clinc_paper_matplotlib_rc = {
    'figure.dpi': 300, 'savefig.dpi': 300,
    'figure.titlesize': 8,
    'lines.linewidth': 0.25,
    'lines.markersize': 2.,
    'patch.linewidth': 0.25,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    "xtick.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 1,
    "axes.labelsize": 5,
    "axes.titlesize": 6,
    "axes.titlepad": 1,
    'axes.linewidth': .25,
    "xtick.labelsize": 5,
    'xtick.major.pad': 1,
    'xtick.major.size': 2,
    'xtick.major.width': 0.25,
    'xtick.minor.pad': .5,
    'xtick.minor.size': 1,
    'xtick.minor.width': 0.25,
    "ytick.labelsize": 5,
    "ytick.major.pad": 1,
    'ytick.major.size': 2,
    'ytick.major.width': 0.25,
    "legend.fontsize": 5,
    "legend.title_fontsize": 6,
    'legend.columnspacing': 0.5,
    'savefig.pad_inches': 0.,
    }

# colors_to_use = sns.color_palette('pastel', n_colors=3) + sns.color_palette('dark', n_colors=3)
clinc_paper_emg_palette = {
    'Right BF': (0.6313725490196078, 0.788235294117647, 0.9568627450980393),
    'Right GAS': (1.0, 0.7058823529411765, 0.5098039215686274),
    'Right EDL': (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
    'Left BF': (0.0, 0.10980392156862745, 0.4980392156862745),
    'Left GAS': (0.6941176470588235, 0.25098039215686274, 0.050980392156862744),
    'Left EDL': (0.07058823529411765, 0.44313725490196076, 0.10980392156862745)}
