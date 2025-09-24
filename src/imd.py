# Intra-Model Difference Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances

# Sort df by predicted classes
def _sort_by_column(df, column):
	df = df.copy(deep=True)
	df["original_index"] = df.index
	return df.sort_values(by=column, ascending=False).reset_index(drop=True)

def plot_classwise_difference(all_models_info, plot_path=None):

	'''
	Input:
	{
		"model_A": {
			"embeddings": [<embed-1>, <embed-2>, ..., <embed-n>],
			"pred_class": [0, 1, 1, 0, ...., 0, 1]
		},
		"model_B": {
			"embeddings": [<embed-1>, <embed-2>, ..., <embed-n>],
			"pred_class": [0, 0, 0, 0, ...., 0, 1]
		},
		...,
		"model_K": {
			"embeddings": [<embed-1>, <embed-2>, ..., <embed-n>],
			"pred_class": [0, 1, 0, 0, ...., 0, 1]
		}
	}
	'''

	# Sort by classes
	all_models_sorted = {}
	for k, v in all_models_info.items():
		all_models_sorted[k] = _sort_by_column(pd.DataFrame(v), "pred_class")
	
	# Create distance matrix and normalize it
	all_models_dist_mat = {}
	for k, v in all_models_sorted.items():
		cs = cosine_distances(v["embeddings"].tolist()).astype(np.float16)
		cs_min = cs.min()
		cs_max = cs.max()
		cs_norm = (cs - cs_min) / (cs_max - cs_min)
		all_models_dist_mat[k] = cs_norm

	########################
	# Heatmap
	########################

	# Determine grid dimensions for subplots
	n_models = len(all_models_sorted)
	cols = 4
	rows = (n_models + cols - 1) // cols

	fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
	axes = axes.flatten()

	# Loop over each model and plot its heatmap in a subplot
	for i, (k, v) in enumerate(all_models_sorted.items()):
		ax = axes[i]
		heatmap = sns.heatmap(
			all_models_dist_mat[k],
			cmap='icefire',
			xticklabels=False,
			yticklabels=False,
			cbar=False,  # disable individual colorbars
			ax=ax)
		ax.invert_yaxis()
		ax.set_title(f"{k}", fontsize=20)
		
	# Remove any unused subplots if the grid has extra spaces.
	for j in range(i + 1, len(axes)):
		fig.delaxes(axes[j])
	
	# Adjust the layout to create space for a common colorbar.
	plt.tight_layout(rect=[0, 0, 0.9, 1])
   
	# Create an axis for the common colorbar on the right side.
	cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
	
	# Create a ScalarMappable with the colormap and the derived vmin and vmax.
	norm = plt.Normalize(vmin=0, vmax=1)
	sm = plt.cm.ScalarMappable(cmap='icefire', norm=norm)
	sm.set_array([])
	fig.colorbar(sm, cax=cbar_ax)
	
	plt.savefig(plot_path or "heatmap.png", format="png", bbox_inches='tight', dpi=300)
	
	return