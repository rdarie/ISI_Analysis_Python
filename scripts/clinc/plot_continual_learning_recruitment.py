import matplotlib as mpl
mpl.use('tkagg')  # generate interactive output
import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import scale, power_transform, minmax_scale
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set(
    context='talk', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    )

folder_path_list = [
    Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312201300-Phoenix"),
    Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202312211300-Phoenix"),
    Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401091300-Phoenix"),
    Path("/users/rdarie/data/rdarie/Neural Recordings/raw/202401111300-Phoenix")
]
normalization_type = 'channel'

if normalization_type == 'channel':
    norm_short_name = 'muscle_norm'
elif normalization_type == 'electrode':
    norm_short_name = 'electrode_norm'
elif normalization_type == 'none':
    norm_short_name = 'no_norm'

routing_info_list = []
for folder_path in folder_path_list:
    this_routing = pd.read_json(folder_path / 'analysis_metadata/routing_config_info.json')
    this_routing['config_start_time'] = this_routing['config_start_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))
    this_routing['config_end_time'] = this_routing['config_end_time'].apply(lambda x: pd.Timestamp(x, tz='GMT'))
    this_routing['folder_path'] = folder_path
    routing_info_list.append(this_routing)

all_routing_info = pd.concat(routing_info_list)
all_routing_info.loc[:, 'exp_day'] = all_routing_info.apply(lambda x: x['config_start_time'].strftime("%y-%m-%d"), axis='columns')

if not os.path.exists(folder_path / "figures"):
    os.makedirs(folder_path / "figures")
pdf_path = folder_path / "figures" / (f'continual_learning_rc_{norm_short_name}.pdf')

per_pulse = False
auc_dict = {}
file_name_suffix = '_per_pulse' if per_pulse else ''
for idx, row in all_routing_info.iterrows():
    file_name = row['child_file_name']
    exp_day = row['exp_day']
    auc_path = row['folder_path'] / (file_name + f'_epoched_emg_auc{file_name_suffix}.parquet')
    print(f'Loading {auc_path}')
    if os.path.exists(auc_path):
        auc_dict[(file_name, exp_day)] = pd.read_parquet(folder_path / auc_path)
    else:
        print('\tWarning! File not found.')

auc_df = pd.concat(auc_dict, names=['block', 'exp_day'])
del auc_dict
'''
fig, ax = plt.subplots()
sns.histplot(auc_df.xs('Left BF', level='channel').reset_index(), x='AUC', ax=ax, palette='pastel', element='step')
plt.show()
'''
# fix 3 outliers in Left BF channel
mask = (auc_df.index.to_frame()['channel'].to_numpy() == 'Left BF') & (auc_df['AUC'].to_numpy() > 0.5)
auc_df = auc_df.loc[~mask, :]


def normalize_by_double(input_df, scaling_fun1, scaling_fun2):
    flat_input = input_df.stack(input_df.columns.names)
    flat_output = scaling_fun1(flat_input.to_numpy().reshape(-1, 1))
    flat_output = pd.Series(scaling_fun2(flat_output).flatten(), index=flat_input.index)
    return flat_output.unstack(input_df.columns.names)


def normalize_by_single(input_df, scaling_fun):
    flat_input = input_df.stack(input_df.columns.names)
    flat_output = scaling_fun(flat_input.to_numpy().reshape(-1, 1))
    flat_output = pd.Series(flat_output.flatten(), index=flat_input.index)
    return flat_output.unstack(input_df.columns.names)


if normalization_type == 'channel':
    rescaled = auc_df.groupby(by='channel', group_keys=False).apply(normalize_by_double, power_transform, minmax_scale)
    data_for_plotting = auc_df.groupby(by='channel', group_keys=False).apply(normalize_by_single, minmax_scale)
elif normalization_type == 'electrode':
    rescaled = auc_df.groupby(by='channel', group_keys=False).apply(normalize_by_single, power_transform)
    rescaled = rescaled.groupby(by='eid', group_keys=False).apply(normalize_by_single, minmax_scale)
    data_for_plotting = auc_df.groupby(by='channel', group_keys=False).apply(normalize_by_single, minmax_scale)
elif normalization_type == 'none':
    rescaled = auc_df.copy()
    data_for_plotting = auc_df.groupby(by='channel', group_keys=False).apply(normalize_by_single, minmax_scale)



def recruitment_radar_plot(
        df=None, ax=None,
        azimuth='channel', azimuth_order=None,
        radius='AUC', hue='StimElec', hue_scales=None,
        color_palette=None, show_titles=False, jitter=0.2, include_scatter=True):
    if azimuth_order is None:
        azimuth_labels = df[azimuth].unique().tolist()
    else:
        azimuth_labels = azimuth_order
    delta_deg = 2 * np.pi / len(azimuth_labels)
    labels_to_degrees = np.arange(0, 2 * np.pi, delta_deg)
    polar_map = {nm: degree for nm, degree in zip(azimuth_labels, labels_to_degrees)}
    # print('\n'.join([f'{key}: {value * 360 / (2 * np.pi):.2f} deg.' for key, value in polar_map.items()]))
    plot_df = df.copy()
    plot_df.loc[:, 'label_as_degree'] = plot_df[azimuth].map(polar_map)
    plot_df.sort_values('label_as_degree', kind='mergesort', inplace=True)

    rng = np.random.default_rng()
    for hue_name, hue_group in plot_df.groupby(hue):
        az_jitter = rng.uniform(-1, 1, hue_group.shape[0]) * delta_deg / 2 * jitter
        average_y = hue_group.groupby('label_as_degree').mean(numeric_only=True).reset_index()
        std_y = hue_group.groupby('label_as_degree').sem(numeric_only=True).reset_index()
        ## wrap around
        average_y = pd.concat([average_y, average_y.iloc[[0], :]])
        std_y = pd.concat([std_y, std_y.iloc[[0], :]])
        if include_scatter:
            ax.plot(
                hue_group['label_as_degree'] + az_jitter, hue_group[radius],
                color=color_palette[hue_name], alpha=0.25,
                marker='o', linewidth=0, markersize=3,
            )
        if hue_scales is not None:
            lw = 1 + 2 * hue_scales[hue_name]
            alpha = 0.5 + 0.5 * hue_scales[hue_name]
        else:
            lw, alpha = 2, 1
        ax.errorbar(
            x=average_y['label_as_degree'], y=average_y[radius], yerr=std_y[radius],
            color=color_palette[hue_name], label=hue_name,
            linewidth=lw, elinewidth=lw, alpha=alpha
        )
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(labels_to_degrees))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(azimuth_labels))
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    return


qlims = [0.05, 0.95]
plot_data = data_for_plotting.reset_index()
plot_data.loc[:, 'AUC'] = plot_data['AUC'] - plot_data['AUC'].min()

freq_labels = plot_data['freq'].unique().tolist()
amp_labels = plot_data['amp'].unique().tolist()

lower_bounds = plot_data.groupby(['freq', 'amp', 'channel']).quantile(qlims[0], numeric_only=True)['AUC']
upper_bounds = plot_data.groupby(['freq', 'amp', 'channel']).quantile(qlims[1], numeric_only=True)['AUC']
y_lims = (lower_bounds.min(), upper_bounds.max())

exp_day_labels = np.unique(plot_data['exp_day']).tolist()
c_palette = sns.cubehelix_palette(start=.5, rot=-.75, n_colors=4)  # sns.color_palette("light:b", n_colors=4)
c_list = [c_palette[idx] for idx in range(len(exp_day_labels))]
c_dict = {ed: c_palette[idx] for idx, ed in enumerate(exp_day_labels)}
ccw_channel_order = ['Right GAS', 'Right BF', 'Left BF', 'Left GAS', 'Left EDL', 'Right EDL']

plot_data.loc[:, 'channel'] = plot_data.apply(lambda x: x['channel'].replace(' ', '\n'), axis='columns')
ccw_channel_order = [nm.replace(' ', '\n') for nm in ccw_channel_order]
group_features = ['eid']
with PdfPages(pdf_path) as pdf:
    for eid, group in plot_data.groupby(group_features):
        print(f'Plotting eid {eid}')
        fig, ax = plt.subplots(
            len(freq_labels), len(amp_labels), figsize=(18, 12),
            sharey=True, subplot_kw=dict(projection='polar'), layout='constrained')
        for name, sub_group in group.groupby(['freq', 'amp']):
            this_freq, this_amp = name
            col_idx, row_idx = freq_labels.index(this_freq), amp_labels.index(this_amp)
            this_ax = ax[col_idx, row_idx]
            recruitment_radar_plot(
                df=sub_group, ax=this_ax,
                azimuth='channel', azimuth_order=ccw_channel_order,
                radius='AUC', hue='exp_day', color_palette=c_dict, include_scatter=False,
                )
            if col_idx == 0:
                this_ax.set_title(f'{int(this_amp)} uA')
            if row_idx == 0:
                this_ax.set_ylabel(f'{int(this_freq)} Hz')
            this_ax.set_ylim(*y_lims)
        this_ax.legend(title='Experiment\nday')
        fig.suptitle(f'eid: {eid}')
        fig.align_labels()
        pdf.savefig(bbox_inches='tight', pad_inches=0.05)
        plt.close()
        # break