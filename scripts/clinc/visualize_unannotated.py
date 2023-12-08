from isicpy.third_party.pymatreader import hdf5todict
from isicpy.utils import makeFilterCoeffsSOS
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate
from pathlib import Path
import h5py
import pandas as pd
import ephyviewer
import numpy as np
from scipy import signal

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

    '''
    folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311071300-Phoenix")
    file_name_list = [
        "MB_1699382682_316178", "MB_1699383052_618936", "MB_1699383757_778055", "MB_1699384177_953948",
        "MB_1699382925_691816", "MB_1699383217_58381", " MB_1699383957_177840"
    ]
    dsi_block_list = []
    file_name = 'MB_1699382682_316178'
    emg_block_name = None
    '''

    # folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
    # file_name = "MB_1699558933_985097"
    # file_name = "MB_1699560317_650555"
    # file_name = 'MB_1699560792_657674'

    # emg_block_name = "Block0002"
    # emg_block_name = "Block0001"

    folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
    file_name_list = [
        'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
        'MB_1700672329_741498', 'MB_1700672668_26337', 'MB_1700673350_780580'
        ]
    file_name = 'MB_1700672668_26337'
    emg_block_name = None

    file_timestamp_parts = file_name.split('_')
    file_start_time = pd.Timestamp(float('.'.join(file_timestamp_parts[1:3])), unit='s', tz='EST')
    t_start_clinc = 0
    if emg_block_name is not None:
        dsi_start_time_str = pd.read_parquet(folder_path / 'dsi_start_times.parquet').loc[emg_block_name, 'start_time']
        dsi_start_time = pd.Timestamp(dsi_start_time_str, tz='EST')
        t_start_dsi = (dsi_start_time - file_start_time).total_seconds()

    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    clinc_trigs = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))

    if emg_block_name is not None:
        emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")
        dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")

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
            dsi_trigs.to_numpy(), dsi_trig_sample_rate, t_start_dsi, channel_names=dsi_trigs.columns)
        trig_view = ephyviewer.TraceViewer(source=dsi_trigs_source, name='emg_trig')
        trig_view.params_controller.on_automatic_color(cmap_name='Set3')

    clinc_source = ephyviewer.InMemoryAnalogSignalSource(
        clinc_df.to_numpy(), clinc_sample_rate, t_start_clinc, channel_names=clinc_df.columns)
    clinc_view = ephyviewer.TraceViewer(source=clinc_source, name='clinc')
    clinc_view.params_controller.on_automatic_color(cmap_name='Set3')
    clinc_spectral_view = ephyviewer.TimeFreqViewer(
        source=clinc_source, name=f'clinc_spectrogram')
    clinc_trig_source = ephyviewer.InMemoryAnalogSignalSource(
        clinc_trigs.to_numpy(), clinc_sample_rate, t_start_clinc, channel_names=clinc_trigs.columns)
    clinc_trig_view = ephyviewer.TraceViewer(source=clinc_trig_source, name='clinc_trigs')
    clinc_trig_view.params_controller.on_automatic_color(cmap_name='Set3')

    win.add_view(clinc_view)
    if emg_block_name is not None:
        win.add_view(emg_signals_view, split_with='clinc', orientation='vertical')
        win.add_view(trig_view, tabify_with='emg')
    win.add_view(clinc_spectral_view, tabify_with='clinc')
    win.add_view(clinc_trig_view, tabify_with='clinc')

    win.show()
    app.exec_()


if __name__ == '__main__':
    visualize_dataset()
