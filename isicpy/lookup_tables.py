import palettable
import pandas as pd
import numpy as np
import seaborn as sns

dsi_channels = {
'PhoenixRight870-2:EMG': 'Right EDL',
'PhoenixRight870-2:EMG.1': 'Right BF',
'PhoenixRight870-2:EMG.2': 'Right GAS',
'PhoenixLeft867-2:EMG': 'Left EDL',
'PhoenixLeft867-2:EMG.1': 'Left BF',
'PhoenixLeft867-2:EMG.2': 'Left GAS',
}

# Col 2:, EMG, mV, Sample Rate: 500.00
# Col 3:, EMG, mV, Sample Rate: 500.00
# Col 4:, EMG, mV, Sample Rate: 500.00
# Col 5:, EMG, mV, Sample Rate: 500.00
# Col 6:, EMG, mV, Sample Rate: 500.00
# Col 7:, EMG, mV, Sample Rate: 500.00

HD64_topo_list = ([
    [-1, -1, 60, 55, 58, 63, -1, -1],
    [24, 54, 47, 46, 53, 52, 59, 25],
    [23, 38, 21, 20, 29, 28, 45, 26],
    [22, 31, 10,  2,  7, 19, 36, 27],
    [32, 30,  0, 13, 16,  9, 37, 35],
    [48, 41, 11,  3,  6, 18, 42, 51],
    [49, 39,  1,  4,  5,  8, 44, 50],
    [56, 40, 12, 14, 15, 17, 43, 57],
    ])
HD64_topo = pd.DataFrame(HD64_topo_list)
HD64_topo.index.name = 'y'
HD64_topo.columns.name = 'x'
HD64_labels = HD64_topo.applymap(lambda x: f"E{x:d}" if (x >= 0) else "")

colors_list = [
    sns.cubehelix_palette(
        n_colors=8, start=st + 1.5, rot=.15, gamma=1., hue=1.0,
        light=0.8, dark=0.1, reverse=False, as_cmap=False)
    for st in np.linspace(0, 3, 10)
    ]
# base_palette = sns.hls_palette(n_colors=8, h=0.01, l=0.6, s=0.65, as_cmap=False)
# colors_list = [
#     sns.hls_palette(n_colors=8, h=0.1, l=l, s=1., as_cmap=False)
#     for l in np.linspace(0.1, 0.8, 8)
#     ]
eids_ordered_xy = HD64_labels.unstack()
eid_palette = {lbl: colors_list[x][y] for (x, y), lbl in eids_ordered_xy.to_dict().items()}

emg_montages = {
    'lower': {
        0: 'LVL',
        1: 'LMH',
        2: 'LTA',
        3: 'LMG',
        4: 'LSOL',
        5: 'L Forearm',
        6: 'RLVL',
        7: 'RMH',
        8: 'RTA',
        9: 'RMG',
        10: 'RSOL',
        11: 'R Forearm',
        12: 'NA',  # not connected
        13: 'NA',  # not connected
        14: 'NA',  # not connected
        15: 'Sync'
    },
    'lower_v2': {
        0: 'LVL',
        1: 'NA',  # not connected
        2: 'LTA',
        3: 'LMG',
        4: 'LSOL',
        5: 'L Forearm',
        6: 'RLVL',
        7: 'RMH',
        8: 'RTA',
        9: 'RMG',
        10: 'RSOL',
        11: 'R Forearm',
        12: 'LMH',
        13: 'NA',  # not connected
        14: 'NA',  # not connected
        15: 'Sync'
    },
    'upper': {
        0: 'L Upper Chest',
        1: 'L Lower Chest',
        2: 'L Side',
        3: 'L Biceps',
        4: 'C Forearm',
        5: 'LTA',
        6: 'R Upper Chest',
        7: 'R L Upper chest',
        8: 'R side',
        9: 'R Biceps',
        10: 'R Forearm',
        11: 'R TA',
        12: 'NA',  # not connected
        13: 'NA',  # not connected
        14: 'NA',  # not connected
        15: 'Sync',
    },
    'ovine_dsi': {
        0: 'LVL',
        1: 'LMH',
        2: 'LTA',
        3: 'LMG',
        4: 'LSOL',
        5: 'L Forearm',
        6: 'RLVL',
        7: 'RMH',
        8: 'RTA',
        9: 'RMG',
        10: 'RSOL',
        11: 'R Forearm',
        12: 'NA',  # not connected
        13: 'NA',  # not connected
        14: 'NA',  # not connected
        15: 'Sync'
    },
}
paired_emg_labels = ['LVL', 'RLVL', 'LMH', 'RMH', 'LTA', 'RTA', 'LMG', 'RMG', 'LSOL', 'RSOL', 'L Forearm', 'R Forearm']
emg_palette = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
emg_hue_map = {
    nm: c for nm, c in zip(paired_emg_labels, emg_palette)
}

muscle_names = {
    'LVL': 'vastus',
    'LMH': 'hamstring',
    'LTA': 'tib. ant.',
    'LMG': 'gastrocnemius',
    'LSOL': 'soleus',
    'L Forearm': 'forearm',
    'RLVL': 'vastus',
    'RMH': 'hamstring',
    'RTA': 'tib. ant.',
    'RMG': 'gastrocnemius',
    'RSOL': 'soleus',
    'R Forearm': 'forearm'
}

# left side only for now
# arm_ctrl_points_epochs = {
#     'Day8_AM':
#         {
#             1: [(659710750, 675300750), (675300750, 696780750), (842000750, 863560750)],
#             2: [(12449750, 27259750), (27259750, 48929750), (460029750, 480869750), (480869750, 503209750), ]
#         }
# }

kinematics_offsets = {
    'Day11_AM': {
        1: 0.,
        2: 0.,
        3: 0.,
        4: 0.,
    },
    'Day12_PM': {
        3: 1.6,
        4: 0.7
    },
    'Day12_AM': {
        2: 0.,
        3: 0.
    },
    'Day8_AM': {
        1: 0.25,
        2: 0.3,
        3: 0.8,
        4: 0.9
    },
    'Day7_AM': {
        4: 0.9
    }
}

video_info = {
    'Day7_AM': {
        4: {
            'paths': [],
            'start_timestamps': [],
            'rollovers': [],
        }
    },
    'Day8_AM': {
        4: {
            'paths': ['/users/rdarie/Desktop/Data Partition Neural Recordings/raw/ISI-C-003/6_Video/Day08_AM_Cam2_GH010042.mp4'],
            'start_timestamps': ['23:56:38:02'],
            'rollovers': [True]
        }
    },
    'Day8_PM': {
        2: {
            'paths': ['/users/rdarie/Desktop/ISI-C-003/6_Video/Day8_PM_GH010626.mp4'],
            'start_timestamps': ['04:17:39:08'],
            'rollovers': [False]
        }
    },
    'Day11_AM': {
        1: {
            'paths': [
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_NDF.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam2_GH010046.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam3_GH010635.mp4',
            ],
            'start_timestamps': ['23:05:24:14', '23:05:56:19', '23:05:19:24'],
            'rollovers': [False, False, False]
        },
        2: {
            'paths': [
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_NDF.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam2_GH010046.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam3_GH010635.mp4',
            ],
            'start_timestamps': ['23:05:24:14', '23:05:56:19', '23:05:19:24'],
            'rollovers': [False, False, False]
        },
        4: {
            'paths': [
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_NDF.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam2_GH010046.mp4',
                '/users/rdarie/Desktop/ISI-C-003/6_Video/Day11_AM_Cam3_GH010635.mp4',
            ],
            'start_timestamps': ['23:05:24:14', '23:05:56:19', '23:05:19:24'],
            'rollovers': [True, True, True]
        }
    },
    'Day12_AM': {
        2: {
            'paths': [],
            'start_timestamps': [],
            'rollovers': [],
        },
        3: {
            'paths': [],
            'start_timestamps': [],
            'rollovers': [],
        }
    },
    'Day12_PM': {
        4: {
            'paths': [
                '/users/rdarie/Desktop/Data Partition Neural Recordings/raw/ISI-C-003/6_Video/Day12_PM_Cam1.mp4',
                '/users/rdarie/Desktop/Data Partition Neural Recordings/raw/ISI-C-003/6_Video/Day12_PM_Cam2_GH010047_Masked.mp4',
                '/users/rdarie/Desktop/Data Partition Neural Recordings/raw/ISI-C-003/6_Video/Day12_PM_Cam3.mp4',
                ],
            'start_timestamps': ['02:36:10:16', '02:17:20:02', '02:17:47:27', ],
            'rollovers': [False, False, False],
            # '/users/rdarie/Desktop/Data Partition Neural Recordings/raw/ISI-C-003/6_Video/Day12_PM_Cam2_2.mp4', '04:02:37:05'
        }
    }
}
