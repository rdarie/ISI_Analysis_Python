import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mne_connectivity import viz

covariance_matrix = np.random.rand(12, 12)
node_names = [f'dummy_{idx}' for idx in range(1, 13)]

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
viz.plot_connectivity_circle(
    covariance_matrix, node_names,
    fig=fig, ax=ax, show=False,
    colormap=sns.color_palette("vlag", as_cmap=True)
    )
fig.savefig('./fake_covariance_matrix.pdf')
plt.show()
