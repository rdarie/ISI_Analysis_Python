from isicpy.third_party.pymatreader import hdf5todict
from pathlib import Path
import h5py
import pandas as pd
import ephyviewer

def visualize_dataset():
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=False)

    folder_path = Path(r"Z:\ISI\Phoenix\202311091300-Phoenix")
    file_name = "MB_1699558933_985097_f.mat"
    file_path = folder_path / "formatted" / file_name

    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    with h5py.File(file_path, 'r') as hdf5_file:
        data = hdf5todict(hdf5_file, variable_names=['data_this_file'], ignore_fields=None)

    emg_df = pd.read_csv(folder_path / "Block0001.ascii", header=12, index_col=0)
    emg_df.index = pd.DatetimeIndex(emg_df.index, tz='EST')
    emg_sample_rate = 500.
    emg_signals = (emg_df - emg_df.mean()).to_numpy()
    t_start = emg_df.index.get_level_values('time_usec')[0] * 1e-6
    emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
        emg_signals, emg_sample_rate, t_start, channel_names=emg_df.columns)
    emg_signals_view = ephyviewer.TraceViewer(source=emg_signals_source, name=f'emg')
    emg_signals_view.params_controller.on_automatic_color(cmap_name='Set3')

