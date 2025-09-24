import umap
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Constants
SEED = 19
REDUCED_DIMENSIONS = 15

# Dimensionality Reduction
def reduce_dimension(embeddings):
	umap_model = umap.UMAP(
		n_components=REDUCED_DIMENSIONS,
		random_state=SEED,
		metric='cosine'
	)
	return umap_model.fit_transform(embeddings)