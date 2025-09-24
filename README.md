# Evaluating Large Language Models for Detecting Antisemitism

<p align="center" width="100%">
    <a href="https://github.com/Ravi2712/quantify-llm-responses/blob/main/assets/paper-teaser.png">
    <img src="https://github.com/Ravi2712/quantify-llm-responses/blob/main/assets/paper-teaser.png" style="width: 450px; max-width: 100%; height: auto; display: block; margin: auto;" alt="paper-teaser" />
    </a>
</p>

## Abstract

Detecting hateful content is a challenging and important problem. Automated tools, like machine‑learning models, can help, but they require continuous training to adapt to the ever-changing landscape of social media. In this work, we evaluate eight open-source LLMs' capability to detect antisemitic content, specifically leveraging in-context definition as a policy guideline. We explore various prompting techniques and design a new CoT-like prompt, Guided-CoT. Guided‑CoT handles the in-context policy well, increasing performance across all evaluated models, regardless of decoding configuration, model sizes, or reasoning capability. Notably, Llama 3.1 70B outperforms fine-tuned GPT-3.5. Additionally, we introduce metrics to quantify semantic divergence in model-generated rationales, revealing notable differences and paradoxical behaviors among LLMs. Our experiments highlight the differences observed across LLMs' utility, explainability, and reliability.

Accepted to EMNLP 2025 Main Conference:

> [**Evaluating Large Language Models for Detecting Antisemitism**](https://arxiv.org/abs/2509.18293),<br/>
[Jay Patel](https://mrjaypatel.com/), [Hrudayangam Mehta](https://www.linkedin.com/in/hrudaymehta/), and [Jeremy Blackburn](https://mrjimmyblack.com). <br>
Binghamton University

------

# 📊 Quantifying LLM Explanations (SCMD & IMD)

When using an LLM as a classifier, you can use the metrics introduced in this paper to quantify LLM-generated explanations for your task. Overall, metrics should be applicable to both binary and multi-class classifications performed through LLM.

## 🚀 Quick Start

### 1) Environment

```bash

Note: The code is tested in a Python 3.10+ base environment.

python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Data layout

Place per-model dataframe files under `sample_data/`:

```
sample_data/
  llama31-70B_q___embeddings.parquet
  qwen-32B___embeddings.parquet
  deepseek-R1-distill-llama33-70B_q___embeddings.parquet
```

Each dataframe must include following columns:

* `response` (optional): LLM-generated response
* `embeddings`: list-of-floats per row (i.e., embedding vector of llm-generated response)
* `pred_class`: integer label per row

### 3) Run

```bash
cd src
python main.py
```

Outputs (by default):

* `scmd_rank.pdf` — model ranking scatter (lower = closer to others) plot
* `scmd_cluster.pdf` — cross-model divergence
* `heatmap.png` — intra-model divergence heatmaps per model (sorted by classes)

## 💡 Metrics

### SCMD — Semantic Cross-Model Divergence

* Summarizes each model via a **Semantic Distance Vector (SDV)**: It is model’s semantic median-distance to every other model.
* Expose two views using SDV, which we call **SCMD**:

  * **Rank**: average median distance over every other model (for a simple leaderboard).
  * **Cluster**: 2D PCA of SDVs to see neighborhoods of models.

### IMD — Intra-Model Differences

* Sorts samples by `pred_class` and visualizes a **within-model cosine-distance heatmap**.
* Useful for spotting tight/loose clusters by class.

## 🧪 Main pipeline notes

Inside `main.py`:

* **UMAP reduction**
  Toggle `is_reduce_dimensions`. When `True`, the code:

  1. converts the raw embedding vector column to `(N, M)` array
  2. call `reduce_dimension(X)` from `utils.py`
  3. **replaces** the `embeddings` column with the reduced dimensions (`(N, d)`)

* **Cross-model metrics**
  `scmd.calculate_semantic_distance_vector({...})` expects:

  ```python
  Input: {
    "model_A": np.ndarray shape (N, d),
    "model_B": np.ndarray shape (N, d),
    ...
  }
  # All arrays must share the SAME shape (N, d). N is number of samples.
  ```

* **Intra-model plots**
  `imd.plot_classwise_difference(...)` expects:

  ```python
  Input: {
    "model_A": [{"embeddings": [...], "pred_class": 1}, ...],
    "model_B": [{"embeddings": [...], "pred_class": 0},...],
    ...
  }
  ```

## 🧭 Usage (Checkout `main.py` file):

```python
# DataFrames per model
all_models_df = {m: df for m, df in ...}

# Calling SCMD:
embeddings_by_model = {
    m: np.array(df["embeddings"].tolist(), dtype=np.float32)
    for m, df in all_models_df.items()
}
sdv = scmd.calculate_semantic_distance_vector(embeddings_by_model)
scmd.semantic_cross_model_divergence_metrics(sdv, kind="rank", plot=True)
scmd.semantic_cross_model_divergence_metrics(sdv, kind="cluster", plot=True)

# Calling IMD:
imd.plot_classwise_difference({		
    m: df[["embeddings", "pred_class"]].to_dict(orient="records")
    for m, df in all_models_df.items()
})
```

## 📄 Citation

```bibtex
@misc{patel2025evaluatinglargelanguagemodels,
      title={Evaluating Large Language Models for Detecting Antisemitism}, 
      author={Jay Patel and Hrudayangam Mehta and Jeremy Blackburn},
      year={2025},
      eprint={2509.18293},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.18293}, 
}
```