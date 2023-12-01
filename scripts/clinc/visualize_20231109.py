from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from pathlib import Path
import h5py
import pandas as pd
import ephyviewer
import numpy as np
from scipy import signal

# clinc_sample_rate = 36931.8
clinc_sample_rate = 36931.71326217823

emg_sample_rate = 500
dsi_trigs_sample_rate = 1000

filterOpts = {
    'low': {
        'Wn': 100.,
        'N': 2,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterCoeffs = makeFilterCoeffsSOS(filterOpts.copy(), emg_sample_rate)
def visualize_dataset():
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=False)

    # folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
    # file_name = "MB_1699558933_985097_f.mat"
    # file_name = "MB_1699560317_650555_f.mat"
    # file_name = 'MB_1699560792_657674_f.mat'

    # emg_block_name = "Block0002"
    # emg_block_name = "Block0001"

    folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
    file_name_list = [
        'MB_1700670158_174163_f.mat', 'MB_1700671071_947699_f.mat', 'MB_1700671568_714180_f.mat',
        'MB_1700672329_741498_f.mat', 'MB_1700672668_26337_f.mat', 'MB_1700673350_780580_f.mat'
    ]
    file_name = 'MB_1700670158_174163_f.mat'
    emg_block_name = "Block0001"

    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')

    clinc_df = pd.read_parquet(folder_path / file_name.replace('.mat', '_clinc.parquet'))
    clinc_df.index = clinc_df.index + file_start_time

    this_routing = {
        'S10': 'E42',
        'S0_S2': 'E59',
        'S14': 'E45',
        'S12_S20': 'E36',
        'S11': 'E37',
        'S16': 'E44',
        'S18': 'E43',
        'S7': 'E48',
        'S1_S3': 'E41',
        'S23': 'E11',
        'S6': 'E3',
        'S22': 'E6',
        'S19': 'E18',
        'S15': 'E51'
    }

    clinc_df.rename(columns=this_routing, inplace=True)
    clinc_trigs = pd.read_parquet(folder_path / file_name.replace('.mat', '_clinc_trigs.parquet'))
    clinc_trigs.index = clinc_trigs.index + file_start_time

    emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")

    dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")

    dsi_coarse_offset = 115
    dsi_fine_offset = 0.6465
    t_zero = file_start_time
    t_start_dsi = (emg_df.index[0] - t_zero).total_seconds() + dsi_coarse_offset + dsi_fine_offset

    t_start_clinc = (file_start_time - t_zero).total_seconds()

    apply_emg_filters = True
    if apply_emg_filters:
        emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
            signal.sosfiltfilt(filterCoeffs, emg_df.to_numpy() ** 2, axis=0),
            emg_sample_rate, t_start_dsi, channel_names=emg_df.columns)
    else:
        emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
            emg_df.to_numpy(),
            emg_sample_rate, t_start_dsi, channel_names=emg_df.columns)

    emg_signals_view = ephyviewer.TraceViewer(source=emg_signals_source, name='emg')
    emg_signals_view.params_controller.on_automatic_color(cmap_name='Set3')

    dsi_trigs_source = ephyviewer.InMemoryAnalogSignalSource(
        dsi_trigs.to_numpy(), dsi_trigs_sample_rate, t_start_dsi, channel_names=dsi_trigs.columns)
    trig_view = ephyviewer.TraceViewer(source=dsi_trigs_source, name='emg_trig')
    trig_view.params_controller.on_automatic_color(cmap_name='Set3')

    clinc_source = ephyviewer.InMemoryAnalogSignalSource(
        clinc_df.to_numpy(), clinc_sample_rate, t_start_clinc, channel_names=clinc_df.columns)
    clinc_view = ephyviewer.TraceViewer(source=clinc_source, name='clinc')
    clinc_view.params_controller.on_automatic_color(cmap_name='Set3')

    clinc_trig_source = ephyviewer.InMemoryAnalogSignalSource(
        clinc_trigs.to_numpy(), clinc_sample_rate, t_start_clinc, channel_names=clinc_trigs.columns)
    clinc_trig_view = ephyviewer.TraceViewer(source=clinc_trig_source, name='clinc_trigs')
    clinc_trig_view.params_controller.on_automatic_color(cmap_name='Set3')

    # custom_path = folder_path / file_name.replace('_f.mat', '_average_zscore.parquet')
    custom_path = None
    if custom_path is not None:
        custom_df = pd.read_parquet(custom_path)
        custom_sample_rate = clinc_sample_rate
        t_start_custom = t_start_clinc
        custom_name = 'average_zscore'

        custom_source = ephyviewer.InMemoryAnalogSignalSource(
            custom_df.to_numpy(), custom_sample_rate, t_start_custom, channel_names=custom_df.columns)
        custom_view = ephyviewer.TraceViewer(source=custom_source, name=custom_name)
        custom_view.params_controller.on_automatic_color(cmap_name='Set3')
    try:
        stim_info = pd.read_parquet(folder_path / file_name.replace('_f.mat', '_stim_info.parquet'))
        pretty_print_fun = lambda x: f'E{x["electrode"]}\namp: {x["amp"]}\nfreq: {x["freq"]}'
        stim_event_dict = {
            'label': stim_info.apply(pretty_print_fun, axis='columns').to_numpy(),
            'time': stim_info.index.total_seconds().to_numpy(),
            'name': f'stim_info'
            }
        event_source = ephyviewer.InMemoryEventSource(all_events=[stim_event_dict])
        event_view = ephyviewer.EventList(source=event_source, name=f'stim_info')
    except Exception:
        pass

    win.add_view(emg_signals_view)
    try:
        win.add_view(event_view, split_with='emg', orientation='horizontal')
    except:
        pass
    win.add_view(clinc_view, split_with='emg', orientation='vertical')
    if custom_path is not None:
        win.add_view(custom_view, split_with='emg', orientation='vertical')
    win.add_view(trig_view, tabify_with='emg')
    win.add_view(clinc_trig_view, tabify_with='clinc')

    win.show()
    app.exec_()


if __name__ == '__main__':
    visualize_dataset()
