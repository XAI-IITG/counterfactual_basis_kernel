# Counterfactuals

# Counterfactual Explanations via Temporal Basis Kernels for Time-Series Models

A framework for generating **counterfactual explanations** on multivariate time-series predictive models using smooth temporal basis functions (B-splines, RBF, Fourier). The project supports multiple real-world datasets and deep-learning architectures for **Remaining Useful Life (RUL) prediction** and **anomaly detection**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Supported Datasets](#supported-datasets)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Generating Counterfactuals](#generating-counterfactuals)
- [Baselines](#baselines)
- [Evaluation Metrics](#evaluation-metrics)
- [Notebooks](#notebooks)
- [Citation](#citation)
- [License](#license)

---

## Overview

Given a trained time-series model $f$ and an input sequence $\mathbf{X} \in \mathbb{R}^{T \times D}$, this framework finds a **counterfactual** $\mathbf{X}_{cf}$ such that:

$$
f(\mathbf{X}_{cf}) \approx y_{\text{target}}, \quad \text{while minimizing } \|\mathbf{X}_{cf} - \mathbf{X}\|
$$

Perturbations are parameterised in a **temporal basis space** $\mathbf{\Phi} \in \mathbb{R}^{T \times K}$ (B-spline, RBF, or Fourier), ensuring smooth, physically plausible changes:

$$
\mathbf{X}_{cf} = \mathbf{X} + \mathbf{\Phi} \mathbf{W}, \quad \mathbf{W} \in \mathbb{R}^{K \times D}
$$

The optimisation jointly minimises **validity** (prediction target), **proximity** (closeness to original), **sparsity**, and **plausibility** losses.

## Key Features

- **Basis-constrained counterfactuals**: B-spline, RBF, and Fourier temporal bases enforce smooth perturbations
- **Multi-dataset support**: C-MAPSS, IEEE PHM 2012, Wind Turbine SCADA, Opportunity UCI, Air Quality India
- **Multiple architectures**: LSTM, GRU, CNN-LSTM, Transformer, TCN
- **Feature schema**: Role-based feature control (action / immutable / state) with MAD-inverse proximity scaling
- **Task flexibility**: Regression (RUL) and classification (anomaly detection) counterfactuals
- **Baseline comparisons**: CoMTE and CoUNTS implementations
- **Comprehensive evaluation**: Validity rate, proximity (L1/L2), sparsity, plausibility, RMSE, MAE, NASA Score

## Project Structure

```
├── src/
│   ├── counterfactuals/    # Core counterfactual generation engine
│   │   ├── core.py         # BasisGenerator — main CF optimiser
│   │   ├── basis.py        # Temporal basis functions (BSpline, RBF, Fourier)
│   │   ├── losses.py       # Multi-objective loss functions
│   │   └── metrics.py      # CF quality metrics
│   ├── baselines/          # CoMTE, CoUNTS baseline explainers
│   ├── data_loader/        # Dataset-specific loaders & preprocessors
│   ├── models/             # Deep learning model definitions per dataset
│   ├── trainer/            # Generic training loop with early stopping
│   ├── evaluation/         # Benchmarking & metrics
│   └── utils/              # Plotting, early stopping, helpers
├── notebooks/              # Exploratory & experiment notebooks per dataset
├── data/                   # Raw & processed datasets
├── outputs/                # Saved models, SHAP results, visualisations
├── main.py                 # Project entry point
└── requirements.txt        # Python dependencies
```

## Supported Datasets

| Dataset | Task | Domain | Reference |
|---------|------|--------|-----------|
| **C-MAPSS** (FD001–FD004) | RUL Regression | Turbofan engine degradation | NASA Prognostics |
| **IEEE PHM 2012** | RUL Regression | Bearing degradation | IEEE PHM Challenge |
| **Wind Turbine SCADA** | Anomaly Detection | Wind turbine fault detection | — |
| **Opportunity UCI** | Activity Classification | Wearable sensor HAR | UCI ML Repository |
| **Air Quality India** | AQI Forecasting | Environmental monitoring | Kaggle |

## Model Architectures

All models are per-dataset and available under `src/models/<dataset>/`:

- **LSTM** — standard sequence model
- **GRU** — gated recurrent unit variant
- **CNN-LSTM** — convolutional feature extraction + LSTM temporal modelling
- **Transformer** — multi-head self-attention encoder

For wind turbine anomaly detection, an **Anomaly Transformer** is used with association discrepancy.

## Installation

```bash
# Clone the repository
git clone https://github.com/XAI-IITG/counterfactual_basis_kernel.git
cd counterfactual_basis_kernel

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch, NumPy, Pandas, scikit-learn, matplotlib, seaborn

## Usage

### Training Models

**C-MAPSS (RUL prediction):**

```bash
python -m src.train_cmapss
```

This trains LSTM, GRU, CNN-LSTM, and Transformer models on the FD001 subset with:
- Sequence length: 50, Max RUL: 125, Batch size: 32
- Low-variance feature removal, z-score normalisation
- Models saved to `outputs/cmapss/FD001/saved_models/`

**IEEE PHM Bearing:**

```bash
python -m src.train_bearing_phm
```

### Generating Counterfactuals

```python
from src.counterfactuals.core import BasisGenerator
from src.counterfactuals.basis import TSFeatureSchema, TargetSpec

# Define feature schema
schema = TSFeatureSchema(
    feature_names=feature_names,
    roles=["action"] * D,       # all features are editable
    min_vals=feature_mins,
    max_vals=feature_maxs,
    mad_inv=mad_inv,
)

# Define target
target = TargetSpec(task_type="regression", target_value=50.0)

# Instantiate generator
gen = BasisGenerator(
    model=model,
    sequence_length=T,
    feature_dim=D,
    basis_type="bspline",   # or "rbf", "fourier"
    num_basis=8,
    device="cuda",
)

# Generate counterfactuals
cfs, info = gen.generate(
    query_instance=x_query,
    target=target,
    schema=schema,
    num_cfs=4,
)
```

See `src/counterfactuals/usage.md` for detailed examples for both regression and classification tasks.

## Baselines

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **CoMTE** | Segment-replacement from training set | `src/baselines/comte.py` |
| **CoUNTS** | Counterfactual via nearest training samples | `src/baselines/counts.py` |

## Evaluation Metrics

**Model performance** (in `src/evaluation/`):
- RMSE, MAE, R²
- NASA Prognostics Score (asymmetric penalty)
- IEEE PHM Score

**Counterfactual quality** (in `src/counterfactuals/metrics.py`):
- **Validity rate**: fraction of CFs reaching target prediction
- **Proximity**: L1 / L2 distance to query instance
- **Sparsity**: fraction of features changed
- **Plausibility**: distance to training data manifold

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/cmapss/005_generate_counterfactuals.ipynb` | End-to-end CF generation for C-MAPSS |
| `notebooks/cmapss/counterfactuals_cmapss_v2.ipynb` | Full-cycle CF with per-unit analysis |
| `notebooks/ieee_phm/phm2012_rul_tcn_transformer_notebook.ipynb` | PHM bearing RUL with TCN/Transformer |
| `notebooks/wind_turbine/02_wind_turbine_counterfactuals.ipynb` | Anomaly Transformer CFs |
| `notebooks/AQI_India/calculating-aqi-air-quality-index-tutorial.ipynb` | AQI computation & data prep |

## Citation

If you use this work, please cite:

```bibtex
@misc{counterfactual_basis_kernel,
  author       = {XAI-IITG},
  title        = {Counterfactual Explanations via Temporal Basis Kernels for Time-Series Models},
  year         = {2025},
  url          = {https://github.com/XAI-IITG/counterfactual_basis_kernel}
}
```

## License

This project is developed at IIT Guwahati. Please refer to the repository for license details.