# Cross-Model Difference Analysis

from itertools import combinations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

'''
Semantic Distance Vector (SDV) is model’s semantic median-distance to every other model.

Input:
{
	"model_A": [<embed-1>, <embed-2>, ..., <embed-n>],
	"model_B": [<embed-1>, <embed-2>, ..., <embed-n>],
	...,
	"model_K": [<embed-1>, <embed-2>, ..., <embed-n>]
}
>>> model_A's value has shape: (N, M)
	- N = Number of data-points
	- M = Embedding dimension

Output:
{
	"model_A": [<dist-to-model-B>, ..., <dist-to-model-K>],
	"model_B": [<dist-to-model-A>, ..., <dist-to-model-K>],
	...,
	"model_K": [<dist-to-model-A>, <dist-to-model-B>, ..., <dist-to-model-(K-1)>],
}
>>> model_A's value has shape: (K,)
	- K = Number of model's to compare against
'''

def calculate_semantic_distance_vector(model_info_dict):

	model_paired_diff = {}
	model_wise_diff = {}
	for m1, m2 in combinations(model_info_dict.keys(), 2):
		X = model_info_dict[m1]
		Y = model_info_dict[m2]
		if X.shape != Y.shape:
			raise ValueError(f"Shape mismatch between {m1} and {m2}: {X.shape} vs {Y.shape}.")
		
		# Calculate cosine-distance matrix
		distance_matrix = cosine_distances(X, Y)
		
		# Normalize matrix to [0,1] range if not already
		if np.any(distance_matrix < 0) or np.any(distance_matrix > 1):
			distance_matrix = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix))

			# Convert to condensed form
			np.fill_diagonal(distance_matrix, 0)

		# Row-wise average
		row_avg = np.mean(distance_matrix, axis=1)
		model_paired_diff[f"{m1}---{m2}"] = row_avg
		
		# Now, model-wise
		if m1 not in model_wise_diff:
			model_wise_diff[m1] = {}
		if m2 not in model_wise_diff:
			model_wise_diff[m2] = {}
			
		if m2 not in model_wise_diff[m1]:
			model_wise_diff[m1].update({
				m2: np.median(row_avg)
			})
		if m1 not in model_wise_diff[m2]:
			model_wise_diff[m2].update({
				m1: np.median(row_avg)
			})

	# Pair-wise Model differences
	# print(model_paired_diff)

	return {k: list(v.values()) for k, v in model_wise_diff.items()}

'''
Now, calculate Semantic Cross-Model Divergence (SCMD) using SDV.
	- We can RANK models: How different their explanations are?
	- We can CLUSTER models: Which models are closer to each other in their explanations?
'''
def semantic_cross_model_divergence_metrics(model_wise_sdv, kind="rank", plot=False, plot_path=None):
	
	models = list(model_wise_sdv.keys())

	if kind == "rank":

		# Compute average median value per model
		average_medians = {m: float(np.mean(model_wise_sdv[m])) for m in models}
		
		# Sort by average value (ascending), so smaller values are on top
		sorted_models = sorted(average_medians.items(), key=lambda x: x[1])
		
		if plot:

			ys = [m for m, _ in sorted_models]
			xs = [v for _, v in sorted_models]

			plt.figure(figsize=(6, max(2, 0.4*len(ys))))
			plt.scatter(xs, ys, s=80)
			for x, y in zip(xs, ys):
				plt.plot([0, x], [y, y], linestyle="--", linewidth=0.8, color="gray")
			plt.xlabel("Average Median Difference")
			plt.ylabel("Model")
			plt.title("SCMD — rank")
			plt.grid(axis="x", linestyle="--", alpha=0.4)
			plt.tight_layout()
			plt.savefig(plot_path or "scmd_rank.pdf", format="pdf", bbox_inches="tight")
			
		return sorted_models

	elif kind == "cluster":

		X = np.array([np.asarray(model_wise_sdv[m], dtype=float).ravel() for m in models])
		
		# Reduce dimensions to 2
		coords = PCA(n_components=2).fit_transform(X)
		result = {m: coords[i] for i, m in enumerate(models)}

		if plot:
			plt.figure(figsize=(5, 5))
			plt.scatter(coords[:, 0], coords[:, 1], s=80)
			for i, m in enumerate(models):
				plt.text(coords[i, 0], coords[i, 1], m, fontsize=8, ha="left", va="bottom")
			plt.xlabel("PC1")
			plt.ylabel("PC2")
			plt.title("SCMD — cluster")
			plt.tight_layout()
			plt.savefig(plot_path or "scmd_cluster.pdf", format="pdf", bbox_inches="tight")
			
		return result