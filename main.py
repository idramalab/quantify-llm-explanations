import numpy as np
import pandas as pd
from tqdm import tqdm

from src import scmd, imd, utils

def main():

	models = ["llama31-70B_q", "qwen-32B", "deepseek-R1-distill-llama33-70B_q"]

	# Load embeddings
	is_reduce_dimensions = True
	all_models_df = {}
	for m in tqdm(models, desc="Loading data files"):
		filename = f"sample_data/{m}___embeddings.parquet"
		df = pd.read_parquet(filename)

		# If True, reduce embedding dimensions using UMAP
		if is_reduce_dimensions:
			X = np.array(df["embeddings"].tolist())
			df["embeddings"] = utils.reduce_dimension(X).tolist()

		# Dict structure {model_A: df1, model_B: df2}
		all_models_df[m] = df

	############################################
	# Cross-Model Differences
	############################################

	sdv = scmd.calculate_semantic_distance_vector({
		m: np.array(df["embeddings"].tolist()) for m, df in all_models_df.items()
	})
	print(f"SDV: {sdv}\n-------")

	# SCMD: Rank
	scmd_rank = scmd.semantic_cross_model_divergence_metrics(sdv, kind="rank", plot=True)
	print(f"SCMD-Rank: {scmd_rank}")
	print(f"Rank plot saved as `scmd_rank.pdf`\n-------")

	# SCMD: Cluster
	scmd_cluster = scmd.semantic_cross_model_divergence_metrics(sdv, kind="cluster", plot=True)
	print(f"SCMD-Cluster: {scmd_cluster}")
	print(f"Cluster plot saved as `scmd_cluster.pdf`\n-------")

	############################################
	# Intra-Model Differences
	############################################

	imd.plot_classwise_difference({		
		m: df[["embeddings", "pred_class"]].to_dict(orient="records") for m, df in all_models_df.items()
	})
	print(f"Heatmap plot saved as `heatmap.png`")

if __name__ == "__main__":
	main()