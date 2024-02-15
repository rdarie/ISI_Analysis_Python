#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering, KMeans
from sklearn.preprocessing import scale, power_transform, minmax_scale
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# get_ipython().run_line_magic('matplotlib', 'tk')

sns.set(
    context='paper', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1, color_codes=True,
    rc={
        'pdf.fonttype': 42,
        'ps.fonttype': 42
        }
)


# In[2]:


num_features = 3
dimensionality_reduction_opts = dict(n_components=num_features, n_neighbors=32)
normalization_type = 'muscle'
electrode_subset = 'midline'
sweep_n_clusters = np.arange(3, 20)

if normalization_type == 'muscle':
    figure_folder = 'normalized_per_muscle'
    norm_short_name = 'muscle_norm'
elif normalization_type == 'electrode':
    figure_folder = 'normalized_per_electrode'
    norm_short_name = 'electrode_norm'
elif normalization_type == 'none':
    figure_folder = 'not_normalized'
    norm_short_name = 'no_norm'

if electrode_subset == 'all':
    subset_suffix = 'all_elec'
elif electrode_subset == 'lateral':
    subset_suffix = 'lateral_elec'
elif electrode_subset == 'midline':
    subset_suffix = 'midline_elec'

num_clus_lookup = {
    ('muscle', 'all'): 5,
    ('muscle', 'lateral'): 4,
    ('muscle', 'midline'): 4,
    ('electrode', 'all'): 7,
    ('none', 'all'): 5,
}
num_clusters = num_clus_lookup[(normalization_type, electrode_subset)]
    
HD64_topo = pd.DataFrame([
    [-1, -1, 60, 55, 58, 63, -1, -1],
    [24, 54, 47, 46, 53, 52, 59, 25],
    [23, 38, 21, 20, 29, 28, 45, 26],
    [22, 31, 10,  2,  7, 19, 36, 27],
    [32, 30,  0, 13, 16,  9, 37, 35],
    [48, 41, 11,  3,  6, 18, 42, 51],
    [49, 39,  1,  4,  5,  8, 44, 50],
    [56, 40, 12, 14, 15, 17, 43, 57],
    ])

class SilhouetteVisualizerModded(SilhouetteVisualizer):

    def fit(self, X, y=None, **kwargs):
        """
        Fits the model and generates the silhouette visualization.
        """
        # Compute the scores of the cluster
        labels = self.estimator.fit_predict(X)
        self.silhouette_score_ = silhouette_score(X, labels)
        self.silhouette_samples_ = silhouette_samples(X, labels)
        # Get the properties of the dataset
        self.n_samples_ = X.shape[0]
        self.n_clusters_ = self.estimator.n_clusters
        # Draw the silhouette figure
        self.draw(labels)
        # Return the estimator
        return self


# In[3]:


raw_data = pd.read_parquet('aucClusteringEachTrial_300ms.parquet')
raw_data.set_index(['StimElec', 'Repeat'], inplace=True)
raw_data.columns = pd.MultiIndex.from_tuples(
    pd.Series(raw_data.columns).apply(lambda x: (x.split('_')[1], int(x.split('_')[3]), int(x.split('_')[5]))).to_list(),
    names=['muscle', 'amp', 'freq'])


# In[4]:


upper_bound = raw_data.stack(level=['muscle', 'amp', 'freq']).quantile(0.99)
'''
fig, ax = plt.subplots()
sns.ecdfplot(data=raw_data.stack(level=['muscle', 'amp', 'freq']), ax=ax)
ax.axvline(upper_bound, color='r')
ax.set_title('Data CDF')
plt.show()
'''
raw_data = raw_data.applymap(lambda x: x if x <= upper_bound else upper_bound)

mask = raw_data.columns.get_level_values('amp') > 500
raw_data = raw_data.loc[:, mask]
'''
mask = raw_data.columns.get_level_values('freq') > 10
raw_data = raw_data.loc[:, mask]
'''

if electrode_subset != 'all':
    if electrode_subset == 'lateral':
        select_eids = HD64_topo.loc[1:, [0, 1, 6, 7]].to_numpy().flatten()
    elif electrode_subset == 'midline':
        select_eids = HD64_topo.loc[:, 2:5].to_numpy().flatten()
    select_mask = raw_data.index.get_level_values('StimElec').astype(int).isin(select_eids)
    raw_data = raw_data.loc[select_mask, :]


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


if normalization_type == 'muscle':
    rescaled = raw_data.groupby(by='muscle', axis='columns').apply(normalize_by_double, power_transform, minmax_scale)
    data_for_plotting = raw_data.groupby(by='muscle', axis='columns').apply(normalize_by_single, minmax_scale)
elif normalization_type == 'electrode':
    rescaled = raw_data.groupby(by='muscle', axis='columns').apply(normalize_by_single, power_transform)
    rescaled = rescaled.groupby(by='StimElec', axis='index').apply(normalize_by_single, minmax_scale)
    data_for_plotting = raw_data.groupby(by='muscle', axis='columns').apply(normalize_by_single, minmax_scale)
elif normalization_type == 'none':
    rescaled = raw_data.copy()
    data_for_plotting = raw_data.groupby(by='muscle', axis='columns').apply(normalize_by_single, minmax_scale)

'''
fig, ax = plt.subplots()
sns.ecdfplot(data=rescaled.stack(level=['muscle', 'amp', 'freq']), ax=ax)
ax.set_title('Rescaled CDF')
plt.show()
'''


# In[5]:


# no dimensionality reduction
'''
features = rescaled.copy()
features.columns = [f'feat_{idx}' for idx in range(raw_data.shape[1])]
'''

# PCA dimensionality reduction
'''
pca = PCA(**dimensionality_reduction_opts)
features = pca.fit_transform(rescaled)
features = pd.DataFrame(
    features, index=raw_data.index,
    columns=[f'feat_{idx}' for idx in range(num_features)])
fig, ax = plt.subplots()
ax.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
'''

# umap for dimensionality reduction
reducer = umap.UMAP(**dimensionality_reduction_opts)
features = reducer.fit_transform(rescaled)

features = pd.DataFrame(
    features, index=raw_data.index,
    columns=[f'feat_{idx}' for idx in range(num_features)])


# In[6]:


'''
fig, ax = plt.subplots(sweep_n_clusters.shape[0] + 1, 1, figsize=(3, 12))
silhouette_score_dict = {}
for idx, n_cl in enumerate(sweep_n_clusters):
    # Instantiate the clustering model and visualizer
    model = SpectralClustering(n_cl)
    visualizer = SilhouetteVisualizerModded(model, colors='sns_pastel', ax=ax[idx])
    visualizer.fit(features)
    visualizer.finalize()        # Finalize and render the figure
    silhouette_score_dict[n_cl] = visualizer.silhouette_score_

silhouette_scores_pd = pd.Series(silhouette_score_dict)
ax[-1].plot(silhouette_scores_pd)
ax[-1].set_xlabel('num clusters')
ax[-1].set_ylabel('silhouette score')
ax[-1].xaxis.set_major_locator(mpl.ticker.FixedLocator(sweep_n_clusters))
'''

fig, ax = plt.subplots()
silhouette_score_dict = {}
for idx, n_cl in enumerate(sweep_n_clusters):
    # Instantiate the clustering model and visualizer
    model = SpectralClustering(n_cl)
    these_labels = model.fit_predict(features)
    silhouette_score_dict[n_cl] = silhouette_score(features, these_labels)

silhouette_scores_pd = pd.Series(silhouette_score_dict)
ax.stem(silhouette_scores_pd.index, silhouette_scores_pd)
ax.set_xlabel('num clusters')
ax.set_ylabel('silhouette score')
ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(sweep_n_clusters))

num_clusters_silhouette = silhouette_scores_pd.idxmax()
print(f'Silhouette analysis recommends {num_clusters_silhouette} clusters')
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_silhouette_analysis.png', bbox_inches='tight', pad_inches=0.05)
plt.close()


# In[7]:


clusterer = SpectralClustering(n_clusters=num_clusters)
# clusterer = HDBSCAN(min_cluster_size=8, min_samples=8)
# clusterer = DBSCAN(eps=.25, min_samples=8)

labels = clusterer.fit_predict(features)
num_found_clusters = np.unique(labels).shape[0]
print(f'Found {num_found_clusters} clusters.')
temp_labeled = raw_data.copy()
temp_labeled['label'] = labels
temp_labeled.set_index('label', inplace=True, append=True)
average_absolute_activation = temp_labeled.stack(temp_labeled.columns.names).groupby(['label']).mean(numeric_only=True)
average_absolute_activation = pd.Series(minmax_scale(average_absolute_activation.to_numpy()), index=average_absolute_activation.index)
new_label_mapping = {old_label: new_label for new_label, (old_label, activation) in enumerate(average_absolute_activation.sort_values(ascending=False).items())}
average_absolute_activation = pd.Series(average_absolute_activation.sort_values(ascending=False).to_numpy())

sorted_labels = [new_label_mapping[old] for old in labels]
labeled_data = raw_data.copy()
labeled_data['label'] = sorted_labels
labeled_data.set_index('label', inplace=True, append=True)

labeled_rescaled = rescaled.copy()
labeled_rescaled.index = labeled_data.index
labeled_features = features.copy()
labeled_features.index = labeled_data.index

if num_features > 2:
    pca_2d = PCA(n_components=2)
    features_2d = pca_2d.fit_transform(features)
    features_2d = pd.DataFrame(features_2d, index=labeled_data.index, columns=[f'feat_{idx}' for idx in range(2)])
else:
    features_2d = labeled_features
    
if num_features > 3:
    pca_3d = PCA(n_components=3)
    features_3d = pca_3d.fit_transform(features)
    features_3d = pd.DataFrame(features_3d, index=labeled_data.index, columns=[f'feat_{idx}' for idx in range(3)])
else:
    features_3d = labeled_features

def max_count(x):
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]

elec_to_label = pd.Series(sorted_labels, index=features.index).groupby('StimElec').apply(max_count)
HD64_E_labels = HD64_topo.applymap(lambda x: f'E{int(x):0>2d}')
HD64_labeled = HD64_topo.applymap(lambda x: elec_to_label.to_dict().get(int(x), -2))

c_palette = sns.color_palette('Set1') + sns.color_palette('Set2')
c_list = [c_palette[idx] for idx in range(num_found_clusters)]
c_dict = {idx: c_palette[idx] for idx in range(num_found_clusters)}


# In[8]:


print(f'plotting {norm_short_name}_{subset_suffix}_embedding_2d...')
text_opts = dict(fontsize='xx-small')
fig, ax = plt.subplots(
    1, 2, width_ratios=[3, 1],
    figsize=(9, 5))
for idx, (lbl, group) in enumerate(features_2d.groupby('label')):
    ax[0].scatter(
        group['feat_0'], group['feat_1'], color=c_dict[lbl],
        label=lbl)

for row_idx, row in features_2d.groupby(['StimElec', 'label']).mean().iterrows():
    stim_elec, label = row_idx[0], row_idx[1]
    ax[0].text(
        row['feat_0'], row['feat_1'],
        f" E{int(stim_elec):0>2d}",
        **text_opts)

ax[0].set_xlabel('Embedding 0')
ax[0].set_ylabel('Embedding 1')
ax[0].xaxis.set_major_formatter(mpl.ticker.NullFormatter())
ax[0].yaxis.set_major_formatter(mpl.ticker.NullFormatter())
ax[0].legend(title='Cluster')

sns.heatmap(
    HD64_labeled, ax=ax[1], cmap=c_list, mask=HD64_labeled<-1,
    xticklabels=False, yticklabels=False, cbar=False,
    annot=HD64_E_labels, annot_kws=text_opts, fmt='s')
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_embedding_2d.png', bbox_inches='tight', pad_inches=0.05)
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_embedding_2d.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


# In[9]:


if num_features > 2:
    print(f'plotting {norm_short_name}_{subset_suffix}_embedding_23...')
    text_opts = dict(fontsize='xx-small')
    fig = plt.figure(figsize=(9, 5), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    for idx, (lbl, group) in enumerate(features_3d.groupby('label')):
        ax.scatter(
            group['feat_0'], group['feat_1'], group['feat_2'],
            color=c_dict[lbl], label=lbl)
    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.zaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.legend(title='Cluster')
    map_ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        HD64_labeled, ax=map_ax, cmap=c_list, mask=HD64_labeled<-1,
        xticklabels=False, yticklabels=False, cbar=False,
        annot=HD64_E_labels, annot_kws=text_opts, fmt='s')
    fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_embedding_3d.png', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_embedding_3d.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()


# In[10]:


def recruitment_radar_plot(
        df=None, ax=None,
        azimuth='muscle', azimuth_order=None,
        radius='signal', hue='StimElec', hue_scales=None,
        color_palette=None, show_titles=False, jitter=0.2, include_scatter=True):
    if azimuth_order is None:
        azimuth_labels = df[azimuth].unique().tolist()
    else:
        azimuth_labels = azimuth_order
    delta_deg = 2 * np.pi / len(azimuth_labels)
    labels_to_degrees = np.arange(0, 2 * np.pi, delta_deg)
    polar_map = {name: degree for name, degree in zip(azimuth_labels, labels_to_degrees)}
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


# In[11]:


qlims = [0.05, 0.95]

print(f'plotting {norm_short_name}_{subset_suffix}_cluster_rc...')
plot_data = labeled_rescaled.stack(level=['muscle', 'amp', 'freq']).to_frame(name='signal').reset_index()
plot_data.loc[:, 'signal'] = plot_data['signal'] - plot_data['signal'].min()

freq_labels = plot_data['freq'].unique().tolist()
amp_labels = plot_data['amp'].unique().tolist()

lower_bounds = plot_data.groupby(['freq', 'amp', 'label']).quantile(qlims[0], numeric_only=True)['signal']
upper_bounds = plot_data.groupby(['freq', 'amp', 'label']).quantile(qlims[1], numeric_only=True)['signal']
y_lims = (lower_bounds.min(), upper_bounds.max())

if normalization_type == 'electrode':
    other_opts = dict(hue_scales=average_absolute_activation)
else:
    other_opts = dict()
    
fig, ax = plt.subplots(
    len(freq_labels), len(amp_labels), figsize=(24, 12),
    sharey=True, subplot_kw=dict(projection='polar'), layout='constrained')

for name, group in plot_data.groupby(['freq', 'amp']):
    this_freq, this_amp = name
    this_ax = ax[freq_labels.index(this_freq), amp_labels.index(this_amp)]
    recruitment_radar_plot(
        df=group, ax=this_ax,
        azimuth='muscle', azimuth_order=[
            'RightGas', 'RightBF', 'LeftBF', 'LeftGas', 'LeftEDL', 'RightEDL'],
        radius='signal', hue='label', color_palette=c_dict, include_scatter=False,
        **other_opts)
    this_ax.set_title(f'{int(this_freq)} Hz {int(this_amp)} uA')
    this_ax.set_ylim(*y_lims)
this_ax.legend(title='Cluster')
fig.align_labels()
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_cluster_rc.png', bbox_inches='tight', pad_inches=0.05)
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_cluster_rc.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


# In[13]:


print(f'plotting {norm_short_name}_{subset_suffix}_electrode_rc...')

if electrode_subset == 'all':
    eids_to_plot = [20, 12, 29]
elif electrode_subset == 'lateral':
    eids_to_plot = [50, 41, 51]
elif electrode_subset == 'midline':
    eids_to_plot = [20, 12, 29]
    
this_plot_data = plot_data.loc[plot_data['StimElec'].astype(int).isin(eids_to_plot), :]

freq_labels = plot_data['freq'].unique().tolist()
amp_labels = plot_data['amp'].unique().tolist()

color_map = sns.color_palette('deep')
c_dict_per_elec = {eid: color_map[idx] for idx, eid in enumerate(eids_to_plot)}

fig, ax = plt.subplots(
    len(freq_labels), len(amp_labels), figsize=(24, 12),
    subplot_kw=dict(projection='polar'))

lower_bounds = this_plot_data.groupby(['freq', 'amp', 'label']).quantile(qlims[0], numeric_only=True)['signal']
upper_bounds = this_plot_data.groupby(['freq', 'amp', 'label']).quantile(qlims[1], numeric_only=True)['signal']
y_lims = (lower_bounds.min(), upper_bounds.max())

for name, group in this_plot_data.groupby(['freq', 'amp']):
    this_freq, this_amp = name
    this_ax = ax[freq_labels.index(this_freq), amp_labels.index(this_amp)]
    recruitment_radar_plot(
        df=group, ax=this_ax,
        azimuth='muscle', radius='signal', hue='StimElec',
        color_palette=c_dict_per_elec)
    this_ax.set_title(f'{int(this_freq)} Hz {int(this_amp)} uA')
    this_ax.set_ylim(*y_lims)
this_ax.legend(title='Electrode')
fig.align_labels()
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_electrode_rc.png', bbox_inches='tight', pad_inches=0.05)
fig.savefig(f'./{figure_folder}/{norm_short_name}_{subset_suffix}_electrode_rc.pdf', bbox_inches='tight', pad_inches=0.05)
plt.close()


# In[12]:


def cluster_illustration_plot(
        df=None, ax=None, page_var=None,
        azimuth='muscle', azimuth_order=None,
        radius='signal', hue='StimElec',
        color_palette=None, show_titles=False, jitter=0.2, include_scatter=True):
    if azimuth_order is None:
        azimuth_labels = df[azimuth].unique().tolist()
    else:
        azimuth_labels = azimuth_order
    delta_deg = 2 * np.pi / len(azimuth_labels)
    labels_to_degrees = np.arange(0, 2 * np.pi, delta_deg)
    polar_map = {name: degree for name, degree in zip(azimuth_labels, labels_to_degrees)}
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
                color=color_palette[page_var], alpha=0.25,
                marker='o', linewidth=0, markersize=3,
                )
        lw, alpha = .5, .25
        ax.errorbar(
            x=average_y['label_as_degree'], y=average_y[radius], yerr=std_y[radius],
            color=color_palette[page_var], # label=hue_name,
            linewidth=lw, elinewidth=lw, alpha=alpha
            )
    
    az_jitter = rng.uniform(-1, 1, plot_df.shape[0]) * delta_deg / 2 * jitter
    average_y = plot_df.groupby('label_as_degree').mean(numeric_only=True).reset_index()
    std_y = plot_df.groupby('label_as_degree').sem(numeric_only=True).reset_index()
    ## wrap around
    average_y = pd.concat([average_y, average_y.iloc[[0], :]])
    std_y = pd.concat([std_y, std_y.iloc[[0], :]])
    if include_scatter:
        ax.plot(
            plot_df['label_as_degree'] + az_jitter, plot_df[radius],
            color=color_palette[page_var], alpha=0.25,
            marker='o', linewidth=0, markersize=3,
            )
    lw, alpha = 2, 1
    ax.errorbar(
        x=average_y['label_as_degree'], y=average_y[radius], yerr=std_y[radius],
        color=color_palette[page_var], # label=hue_name,
        linewidth=lw, elinewidth=lw, alpha=alpha
        )
    
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(labels_to_degrees))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(azimuth_labels))
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    if show_titles:
        ax.set_title(df[ax_var].unique()[0])
    return


# In[14]:


print(f'plotting {norm_short_name}_{subset_suffix}_cluster_composition...')
plot_data = labeled_rescaled.stack(level=['muscle', 'amp', 'freq']).to_frame(name='signal').reset_index()
plot_data.loc[:, 'signal'] = plot_data['signal'] - plot_data['signal'].min()

freq_labels = plot_data['freq'].unique().tolist()
amp_labels = plot_data['amp'].unique().tolist()
pdf_path = f'./{figure_folder}/{norm_short_name}_{subset_suffix}_cluster_composition.pdf'

with PdfPages(pdf_path) as pdf:
    for cluster, group in plot_data.groupby('label'):
        print(f'on cluster {cluster}')
        lower_bounds = group.groupby(['freq', 'amp', 'StimElec']).quantile(qlims[0], numeric_only=True)['signal']
        upper_bounds = group.groupby(['freq', 'amp', 'StimElec']).quantile(qlims[1], numeric_only=True)['signal']
        y_lims = (lower_bounds.min(), upper_bounds.max())

        fig, ax = plt.subplots(
            len(freq_labels), len(amp_labels), figsize=(12, 12),
            sharey=True, subplot_kw=dict(projection='polar'), layout='constrained')
        for name, subgroup in group.groupby(['freq', 'amp']):
            this_freq, this_amp = name
            this_ax = ax[freq_labels.index(this_freq), amp_labels.index(this_amp)]
            cluster_illustration_plot(
                df=subgroup, ax=this_ax, page_var=cluster,
                azimuth='muscle', azimuth_order=[
                    'RightGas', 'RightBF', 'LeftBF', 'LeftGas', 'LeftEDL', 'RightEDL'],
                radius='signal', hue='StimElec', color_palette=c_dict, include_scatter=False
                )
            this_ax.set_title(f'{int(this_freq)} Hz {int(this_amp)} uA')
            this_ax.set_ylim(*y_lims)
        fig.suptitle(f'Cluster {cluster}')
        fig.align_labels()
        pdf.savefig(bbox_inches='tight', pad_inches=0.05)
        plt.close()


# In[ ]:





# In[ ]:




