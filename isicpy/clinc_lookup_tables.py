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
    '202311221100-Phoenix': -12,
    '202312080900-Phoenix': -6,
}
