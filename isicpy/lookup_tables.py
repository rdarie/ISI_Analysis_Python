import palettable

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
