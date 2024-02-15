import pandas as pd


def parse_mb_stim_csv(file_path):
    par = pd.read_csv(file_path, header=1, index_col=0)
    par.index.name = 'cactus_chan'
    par.columns = [
        'amp', 'pw', 'DZ0', 'DELTA', 'DZ1',
        'M', 'DZ2', 'THP_DEL', 'N', 'DZ3', 'P', 'RAMP_SCALE_OPT'
        ]
    null_mask = (par == 0).all(axis=1)
    par = par.loc[~null_mask, :].astype(int)

    par['amp'] = par['amp'] * 12  # uA
    par['pw'] = par['pw'] * 10  # usec
    par['DZ0'] = par['DZ0'] * 10  # usec
    par['DELTA'] = par['DELTA'] * 10  # usec
    par['DZ1'] = par['DZ1'] * 80  # usec
    par['DZ2'] = par['DZ2'] * 40.96  # msec
    par['THP_DEL'] = par['THP_DEL'] * 40.96  # msec
    par['DZ3'] = par['DZ3'] * 20.97  # sec

    def parse_one_row(row):
        if row['M'] > 1:
            freq = (row['DZ1'] * 1e-6) ** -1
            train_dur = (row['M'] - 1) * row['DZ1'] * 1e-6
            train_period = row['DZ2'] * 1e-3
        else:
            freq = (row['DZ2'] * 1e-3) ** -1
            train_dur = (row['N'] - 1) * row['DZ2'] * 1e-3
            train_period = row['DZ3']
        return freq, train_dur, train_period

    par[['freq', 'train_dur', 'train_period']] = pd.DataFrame(
        par.apply(parse_one_row, axis='columns').tolist(),
        index=par.index,
        columns=['freq', 'train_dur', 'train_period'])

    return par


def assign_stim_metadata(t, stim_dict_list=None):
    for row in stim_dict_list:
        if (t > row['start_time'].total_seconds()) & (t < row['end_time'].total_seconds()):
            return row['params']
    return None

def assign_tens_metadata(t, stim_dict_list=None):
    for row in stim_dict_list:
        if (t > row['start_time']) & (t < row['end_time']):
            output = {
                key: row[key]
                for key in ['location', 'amp', 'pw']
            }
            return pd.Series(output)
    return None
