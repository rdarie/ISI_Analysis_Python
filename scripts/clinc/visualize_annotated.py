
from isicpy.utils import makeFilterCoeffsSOS
from isicpy.clinc_lookup_tables import clinc_sample_rate, emg_sample_rate, dsi_trig_sample_rate, dsi_mb_clock_offsets
from pathlib import Path
from isicpy.lookup_tables import dsi_channels
import pandas as pd
import ephyviewer
import numpy as np
from scipy import signal
import json
import os


filterOptsEmg = {
    'low': {
        'Wn': 100.,
        'N': 2,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterCoeffsEmg = makeFilterCoeffsSOS(filterOptsEmg.copy(), emg_sample_rate)

filterOptsClinc = {
    'low': {
        'Wn': 1000.,
        'N': 8,
        'btype': 'low',
        'ftype': 'butter'
    }
}
filterCoeffsClinc = makeFilterCoeffsSOS(filterOptsClinc.copy(), clinc_sample_rate)

apply_emg_filters = False
apply_clinc_filters = True
apply_custom_filters = True
show_clinc_spectrogram = True
show_custom_spectrogram = True
def visualize_dataset():
    app = ephyviewer.mkQApp()
    win = ephyviewer.MainViewer(debug=False)

    # folder_path = Path(r"/users/rdarie/data/rdarie/Neural Recordings/raw/20231109-Phoenix")
    # file_name_list = ["MB_1699558933_985097", "MB_1699560317_650555", 'MB_1699560792_657674']
    # file_name = "MB_1699558933_985097"

    # emg_block_name = "Block0002"
    # emg_block_name = "Block0001"

    folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202311221100-Phoenix")
    file_name_list = [
        'MB_1700670158_174163', 'MB_1700671071_947699', 'MB_1700671568_714180',
        # 'MB_1700672329_741498',
        'MB_1700672668_26337', 'MB_1700673350_780580'
        ]
    file_name = 'MB_1700672668_26337'

    folder_path = Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312080900-Phoenix")
    file_name_list = [
        "MB_1702047397_450767", "MB_1702048897_896568", "MB_1702049441_627410",
        "MB_1702049896_129326", "MB_1702050154_688487", "MB_1702051241_224335"
    ]
    file_name = 'MB_1702049896_129326'

    if os.path.exists(folder_path / 'dsi_block_lookup.json'):
        with open(folder_path / 'dsi_block_lookup.json', 'r') as f:
            emg_block_name = json.load(f)[file_name][0]
    else:
        emg_block_name = None

    # custom_name = 'average_zscore'
    custom_name = 'clinc_reref'
    filterCoeffsCustom = filterCoeffsClinc


    stim_info_file = (file_name + '_tens_info.parquet')
    custom_path = folder_path / (file_name + f'_{custom_name}.parquet')

    clinc_df = pd.read_parquet(folder_path / (file_name + '_clinc.parquet'))
    clinc_trigs = pd.read_parquet(folder_path / (file_name + '_clinc_trigs.parquet'))
    t_start_clinc = 0

    if emg_block_name is not None:
        clock_difference = dsi_mb_clock_offsets[folder_path.stem]
        with open(folder_path / 'dsi_to_mb_fine_offsets.json', 'r') as f:
            dsi_fine_offset = json.load(f)[file_name][emg_block_name]
        dsi_total_offset = pd.Timedelta(clock_difference + dsi_fine_offset, unit='s')
        print(f'DSI offset = {clock_difference} + {dsi_fine_offset:.3f} = {dsi_total_offset.total_seconds():.3f}')
        emg_df = pd.read_parquet(folder_path / f"{emg_block_name}_emg.parquet")
        emg_df.rename(columns=dsi_channels, inplace=True)
        dsi_trigs = pd.read_parquet(folder_path / f"{emg_block_name}_dsi_trigs.parquet")  # .iloc[:, [1]]
        dsi_trigs.index = dsi_trigs.index + dsi_total_offset

        t_start_dsi = (dsi_trigs.index[0] - clinc_df.index[0]).total_seconds()

        if apply_emg_filters:
            emg_signals_source = ephyviewer.InMemoryAnalogSignalSource(
                signal.sosfiltfilt(filterCoeffsEmg, emg_df.to_numpy() ** 2, axis=0),
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

    if apply_clinc_filters:
        clinc_source = ephyviewer.InMemoryAnalogSignalSource(
            signal.sosfiltfilt(filterCoeffsClinc, clinc_df.to_numpy(), axis=0),
            clinc_sample_rate, t_start_clinc, channel_names=clinc_df.columns)
    else:
        clinc_source = ephyviewer.InMemoryAnalogSignalSource(
            clinc_df.to_numpy(),
            clinc_sample_rate, t_start_clinc, channel_names=clinc_df.columns)
    clinc_view = ephyviewer.TraceViewer(source=clinc_source, name='clinc')
    clinc_view.params_controller.on_automatic_color(cmap_name='Set3')
    if show_clinc_spectrogram:
        clinc_spectral_view = ephyviewer.TimeFreqViewer(
            source=clinc_source, name=f'clinc_spectrogram')
    clinc_trig_source = ephyviewer.InMemoryAnalogSignalSource(
        clinc_trigs.to_numpy(), clinc_sample_rate, t_start_clinc, channel_names=clinc_trigs.columns)
    clinc_trig_view = ephyviewer.TraceViewer(source=clinc_trig_source, name='clinc_trigs')
    clinc_trig_view.params_controller.on_automatic_color(cmap_name='Set3')

    if custom_path is not None:
        custom_df = pd.read_parquet(custom_path)
        custom_sample_rate = clinc_sample_rate
        t_start_custom = t_start_clinc

        if apply_custom_filters:
            custom_source = ephyviewer.InMemoryAnalogSignalSource(
                signal.sosfiltfilt(filterCoeffsCustom, custom_df.to_numpy(), axis=0),
                custom_sample_rate, t_start_custom, channel_names=custom_df.columns)
        else:
            custom_source = ephyviewer.InMemoryAnalogSignalSource(
                custom_df.to_numpy(),
                custom_sample_rate, t_start_custom, channel_names=custom_df.columns)
        custom_view = ephyviewer.TraceViewer(source=custom_source, name=custom_name)
        custom_view.params_controller.on_automatic_color(cmap_name='Set3')
        if show_custom_spectrogram:
            custom_spectral_view = ephyviewer.TimeFreqViewer(
                source=custom_source, name=f'{custom_name}_spectrogram')

    stim_info_full_path = folder_path / stim_info_file
    if os.path.exists(stim_info_full_path):
        stim_info = pd.read_parquet(stim_info_full_path)
        stim_info = stim_info.reset_index()
        stim_info['timestamp'] = stim_info['timestamp'] - clinc_df.index[0]
        if 'per_pulse' in stim_info_file:
            pretty_print_fun = lambda x: f'\nE{x["eid"]}\namp: {int(x["amp"])}\nfreq: {x["freq"]:.2f}\nrank: {x["rank_in_train"]}'
        elif 'tens' in stim_info_file:
            pretty_print_fun = lambda x: f'\nTENS {x["location"]}\namp: {int(x["amp"])}'
        else:
            pretty_print_fun = lambda x: f'\nE{x["eid"]}\namp: {int(x["amp"])}\nfreq: {x["freq"]:.2f}'
        stim_event_dict = {
            'label': stim_info.apply(pretty_print_fun, axis='columns').to_numpy(),
            'time': np.asarray([ts.total_seconds() for ts in stim_info['timestamp']]),
            'name': f'stim_info'
            }
        event_source = ephyviewer.InMemoryEventSource(all_events=[stim_event_dict])
        event_view = ephyviewer.EventList(source=event_source, name=f'stim_info')

    win.add_view(clinc_view)
    if os.path.exists(stim_info_full_path):
        win.add_view(event_view, split_with='clinc', orientation='horizontal')
    if custom_path is not None:
        win.add_view(custom_view, split_with='clinc', orientation='vertical')
    if show_custom_spectrogram:
        win.add_view(custom_spectral_view, tabify_with=custom_name)
    if emg_block_name is not None:
        win.add_view(emg_signals_view, split_with='clinc', orientation='vertical')
        win.add_view(trig_view, tabify_with='emg')
    if show_clinc_spectrogram:
        win.add_view(clinc_spectral_view, tabify_with='clinc')
    win.add_view(clinc_trig_view, tabify_with='clinc')

    win.show()
    app.exec_()


if __name__ == '__main__':
    visualize_dataset()
