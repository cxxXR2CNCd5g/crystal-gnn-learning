# Crystal GNN for Formation Energy Prediction

A beginner-friendly crystal graph neural network project for predicting **formation energy per atom** from CIF crystal structures using **PyTorch** and **PyTorch Geometric**.

This repository records my learning process from CIF parsing and crystal graph construction to model training, evaluation, and result visualization.

> This is an educational baseline implementation rather than a state-of-the-art crystal graph model.

---

## Project Overview

The complete workflow is:

```text
Materials Project API
        ↓
Download CIF structures and formation_energy_per_atom
        ↓
Read crystal structures with ASE
        ↓
Build atomic node features
        ↓
Construct edges using a cutoff radius
        ↓
Expand atomic distances with Gaussian basis functions
        ↓
Save crystal graphs as PyTorch Geometric Data objects
        ↓
Train a crystal graph neural network
        ↓
Evaluate and visualize prediction results
```

The current model uses:

- Atomic-number one-hot encoding as node features
- A 4.0 Å cutoff radius for neighbor construction
- Gaussian expansion of interatomic distances
- Two `CGConv` layers
- Global mean pooling
- A multilayer perceptron for regression

The prediction target is `formation_energy_per_atom`, with units of `eV/atom`.

---

## Current Result

The model was trained on 20,000 crystal structures obtained from Materials Project.

| Dataset split | Number of structures |
|---|---:|
| Training set | 16,000 |
| Validation set | 2,000 |
| Test set | 2,000 |

Current test performance:

| Metric | Value |
|---|---:|
| Test MSE | 0.01634 |
| Test MAE | 0.07953 eV/atom |

The current result is intended as a reproducible baseline for further study.

---

## Repository Structure

```text
learn_gnn/
├── data_download/
│   ├── get_MP.py
│   └── mp-ids-46744.csv
│
├── scripts/
│   ├── crystal_gnn/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   ├── dataset.py
│   │   ├── data_loader.py
│   │   └── model.py
│   │
│   ├── 01_build_single_data.py
│   ├── 1single_graph_train_step.py
│   ├── 02_build_full_dataset.py
│   ├── 03_build_dataloader.py
│   ├── 04_prepare_dataloaders.py
│   ├── 05_train.py
│   └── 06_visualize_results.py
│
├── checkpoint/
├── results_20000/
├── .gitignore
└── README.md
```

Large generated datasets and PyTorch `.pt` files are not included in the repository.

---

## Environment

Recommended environment:

- Python 3.10 or 3.11
- PyTorch
- PyTorch Geometric
- ASE
- pandas
- NumPy
- matplotlib
- mp-api

Install the main dependencies with:

```bash
pip install torch
pip install torch-geometric
pip install ase pandas numpy matplotlib mp-api
```

For CUDA installation, install a PyTorch version compatible with your CUDA environment.

---

## Materials Project Data

Crystal structures and target values are obtained through the Materials Project API.

The API returns:

- Crystal structure information
- Materials Project ID
- `formation_energy_per_atom`

Set your Materials Project API key as an environment variable:

```bash
export MP_API_KEY="your_api_key"
```

Do not write a real API key directly into public source code.

Then run:

```bash
python data_download/get_MP.py
```

The raw CIF data are not uploaded to GitHub because of file size.

---

## Crystal Graph Construction

Each CIF structure is converted into a PyTorch Geometric `Data` object.

The graph contains:

```text
x             node features
z             atomic numbers
edge_index    graph connectivity
edge_attr     Gaussian-expanded distance features
y             formation energy per atom
structure_id  structure identifier
```

### Node Features

The current implementation uses a 100-dimensional one-hot vector based on atomic number.

```python
x = one_hot(atomic_number)
```

### Edge Construction

Pairwise atomic distances are calculated with the periodic minimum-image convention.

Atoms are connected when:

```text
0 < distance ≤ cutoff radius
```

The current cutoff radius is:

```text
4.0 Å
```

### Gaussian Distance Expansion

A scalar distance is converted into a continuous multidimensional edge feature:

\[
e_{ij}^{(k)}
=
\exp\left[
-rac{(d_{ij}-\mu_k)^2}{\sigma^2}
ight]
\]

where:

- \(d_{ij}\) is the distance between atoms \(i\) and \(j\)
- \(\mu_k\) is the center of the \(k\)-th Gaussian basis
- \(\sigma\) controls the Gaussian width

With a cutoff of 4.0 Å and an interval of 0.2 Å, each edge is represented by a 21-dimensional feature vector.

---

## Running the Project

Run the scripts in the following order.

### 1. Build and inspect one crystal graph

```bash
python scripts/01_build_single_data.py
```

### 2. Test one training step

```bash
python scripts/1single_graph_train_step.py
```

### 3. Preprocess all CIF structures

```bash
python scripts/02_build_full_dataset.py
```

This converts CIF structures into separate `.pt` graph files.

### 4. Test the DataLoader

```bash
python scripts/03_build_dataloader.py
```

### 5. Prepare train, validation, and test loaders

```bash
python scripts/04_prepare_dataloaders.py
```

### 6. Train the model

```bash
python scripts/05_train.py
```

The best model is saved according to validation MSE.

Expected output files include:

```text
checkpoint/best_model_20000.pt

results_20000/
├── training_history.csv
├── test_predictions.csv
└── test_metrics.txt
```

### 7. Visualize results

```bash
python scripts/06_visualize_results.py
```

The visualization script generates:

```text
results_20000/figures/
├── 01_training_curve.png
├── 02_test_parity_plot.png
├── 03_absolute_error_distribution.png
└── 04_residual_plot.png
```

---

## Model Architecture

The current baseline model is:

```text
Atomic one-hot features
        ↓
Linear layer: 100 → 64
        ↓
CGConv layer
        ↓
CGConv layer
        ↓
Global mean pooling
        ↓
MLP regression head
        ↓
Formation energy per atom
```

The saved model file mainly contains trained parameter tensors, including weight matrices, bias vectors, and batch-normalization states.

---

## Current Limitations

The current implementation is intentionally simple.

Important limitations include:

- Node features only contain atomic-number one-hot encoding
- Graph construction mainly uses pairwise distances
- Bond-angle and higher-order geometric information are not included
- Periodic image offsets are not explicitly stored
- The dataset is randomly split
- Structurally similar materials may appear in different subsets
- The model has not been compared systematically with stronger baselines

Therefore, the current result should be regarded as a learning baseline rather than a final research model.

---

## Future Work

Planned improvements include:

- Replace one-hot encoding with trainable element embeddings
- Add atomic and structural descriptors
- Improve periodic neighbor construction
- Add coordination numbers and bond-angle information
- Introduce three-body interactions
- Study graph attention mechanisms
- Compare different pooling strategies
- Add learning-rate scheduling and early stopping
- Evaluate stricter dataset splitting methods
- Compare with models such as ALIGNN, M3GNet, CHGNet, and equivariant graph networks

Each modification will be tested independently so that its effect on MAE, RMSE, and \(R^2\) can be evaluated clearly.

---

## Purpose of This Repository

The main purpose of this project is to understand the complete path from a crystal structure file to a graph neural network prediction:

```text
CIF
→ atomic structure
→ crystal graph
→ graph batch
→ message passing
→ crystal representation
→ property prediction
```

This repository will continue to be updated as I study more advanced crystal graph construction and message-passing methods.

---

## License

This project can be released under the MIT License.
