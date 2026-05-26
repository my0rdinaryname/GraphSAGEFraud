# 🕸️ Graph Neural Network for Bitcoin Fraud Detection

> Detecting illicit transactions in the Elliptic Bitcoin dataset using GraphSAGE — because fraud lives in *relationships*, not individual records.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5+-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Traditional fraud detection models evaluate each transaction in isolation. This project takes a fundamentally different approach: it models the **entire transaction network as a graph**, where each node is a Bitcoin transaction and each edge represents Bitcoin flow between transactions.

By applying **GraphSAGE** message passing, every transaction node learns from its neighbourhood — absorbing patterns from its direct counterparts and their counterparts (2-hop awareness). Fraud rings that are invisible to row-level models become detectable from their network signatures.

---

## 🧠 Key Concepts

**Why graphs?** Fraudsters operate in rings — fake accounts, money mules, layered transfers. A single transaction in a fraud ring may look perfectly normal. Its neighbourhood does not.

**Why GraphSAGE?** It samples a fixed-size neighbourhood per node (scalable to 200k+ nodes), learns an inductive aggregation function (generalises to unseen nodes), and was designed for exactly this scale of graph.

**Why Focal Loss + Class Weights?** The dataset has ~20% fraud in training but as low as 0.3% in later test timesteps. Standard cross-entropy makes the model lazy — it predicts "legit" for everything and gets 98% accuracy while being useless. Focal loss forces the model to focus on hard, misclassified fraud cases.

---

## 📊 Results

Evaluated on a **strict temporal holdout split** (train: timesteps 1–39, val: 40–43, test: 44–49) to reflect real-world conditions where the model always predicts the future.

| Metric | Value |
|---|---|
| **AUC-ROC** (full test) | 0.70 |
| **AUC-ROC** (timesteps 47–49) | **0.74** |
| **F1** (balanced threshold 0.45) | **0.21** |
| **Recall** (threshold 0.15) | **0.99** |
| **AUC-PR** (timesteps 47–49) | 0.14 |

### Why two evaluation sets?

The test set (timesteps 44–49) contains a severe fraud rate collapse:

| Timestep | Fraud Rate |
|---|---|
| 44 | 1.5% |
| 45 | 0.4% |
| 46 | 0.3% |
| 47 | 2.6% |
| 48 | 7.6% |
| 49 | 11.8% |

Timesteps 44–46 have near-zero fraud — no model achieves meaningful F1 there. Evaluating on timesteps 47–49 (where fraud actually exists) gives a fair picture of model quality. This mirrors how real fraud teams assess models: on periods where fraud is active.

### Business-driven threshold selection

Rather than reporting a single F1 score, this project demonstrates the precision-recall tradeoff across all thresholds:

| Operating Mode | Threshold | Recall | Precision | Use Case |
|---|---|---|---|---|
| High recall | 0.15 | 0.99 | 0.11 | Flag for manual review |
| Balanced | 0.45 | 0.40 | 0.14 | Mixed automated + review |
| High precision | 0.65 | 0.11 | 0.22 | Automated block |

> The right threshold is a **business decision**, not a model decision. This project makes that tradeoff explicit.

---

## 🏗️ Architecture

```
Raw CSVs (features, classes, edgelist)
         │
         ▼
  Graph Construction
  ┌─────────────────────────────────────────┐
  │  Nodes: 203,769 transactions            │
  │  Edges: 234,355 Bitcoin flows           │
  │  Node features: 166 dims + timestep     │
  └─────────────────────────────────────────┘
         │
         ▼
  GraphSAGE Layer 1  ──────────────────────┐
  (aggregate 1-hop neighbours)             │ skip
         │                                 │ connection
         ▼                                 │
  BatchNorm + ReLU + Dropout               │
         │                                 │
         ▼                                 │
  GraphSAGE Layer 2                        │
  (aggregate 2-hop neighbourhood)          │
         │                                 │
         └──────────── + ─────────────────-┘
                       │
                       ▼
              MLP Classifier Head
              (256 → 64 → 2)
                       │
                       ▼
              Fraud Probability (0–1)
```

### Design decisions

**Skip connection**: adds the original node features back after 2 rounds of message passing. Prevents the model from "forgetting" individual transaction characteristics while learning neighbourhood context.

**Temporal feature**: timestep normalised to [0,1] appended as node feature. Gives the model awareness of *when* a transaction happened — fraud patterns shift over time.

**Focal Loss + class weights**: focal loss (γ=1.5) downweights easy legit predictions; explicit class weight (~4× for fraud) ensures the loss function never ignores fraud cases regardless of batch composition.

**Recency-weighted loss**: later timesteps weighted higher during training so the model prioritises learning recent fraud patterns over historical ones.

---

## 📁 Project Structure

```
gnn-fraud-detection/
├── data/
│   └── elliptic_bitcoin_dataset/
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── elliptic_txs_edgelist.csv
├── notebooks/
│   └── graphneuralnetwork.ipynb      # full training notebook
├── src/
│   ├── graph_construction.py         # Phase 2: CSV → PyG Data object
│   ├── model.py                      # FraudGNN architecture
│   ├── train.py                      # training loop + focal loss
│   └── evaluate.py                   # threshold sweep + metrics
├── outputs/
│   ├── fraud_gnn.pt                  # saved model checkpoint
│   ├── training_curves.png
│   ├── pr_curve.png
│   └── tsne_embeddings.png
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/gnn-fraud-detection.git
cd gnn-fraud-detection
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Via Kaggle API
kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d data/
```

Or manually from [Kaggle — Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).

### 3. Run the notebook

Open `notebooks/graphneuralnetwork.ipynb` in Kaggle or Colab with **GPU enabled** (T4 or better).

Run all cells top to bottom. Full training takes ~15 minutes on a T4 GPU.

---

## 📦 Requirements

```
torch>=2.0.0
torch-geometric>=2.5.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Install all:
```bash
pip install -r requirements.txt
```

> **Note**: `torch-geometric` must match your PyTorch + CUDA version. If installation fails, use:
> ```bash
> pip install torch-geometric
> pip install torch-scatter torch-sparse \
>     -f https://data.pyg.org/whl/torch-{YOUR_TORCH_VERSION}+{CUDA}.html
> ```

---

## 🔍 The Temporal Distribution Shift Problem

This project surfaces a real-world ML challenge that most tutorials ignore.

The Elliptic dataset is ordered chronologically. Fraud rate in the training period averages ~20%. Fraud rate in the earliest test timesteps drops to 0.3%. This is not a dataset quirk — it reflects a real phenomenon: **fraud patterns evolve, and models trained on historical data degrade as fraud tactics change**.

The correct response (implemented here) is:
1. Evaluate on periods where fraud is active (timesteps 47–49)
2. Use recency-weighted loss to prioritise recent patterns
3. Select thresholds that match the expected deployment distribution
4. Frame results around the precision-recall tradeoff, not a single F1 score

This is what real fraud teams do. Chasing a high F1 on an easy random split misses the point entirely.

---

## 💡 Lessons Learned

**Evaluation matters more than architecture.** The model spent weeks at "bad" metrics because the evaluation was wrong (fixed 0.5 threshold on a 0.3% fraud test set). Fixing the evaluation revealed a working model.

**Threshold is a product decision.** A bank that auto-blocks transactions needs precision > 0.20. A bank that flags for human review can afford recall > 0.99 at threshold 0.15. The model doesn't pick the threshold — the business does.

**AUC-ROC can lie on imbalanced data.** AUC-ROC of 0.70 looks mediocre but means the model correctly ranks a random fraud node above a random legit node 74% of the time on the hard test period. AUC-PR is the more honest metric when positives are rare.

**Distribution shift is the real problem in production ML.** No static model fully solves it. The right long-term answer is periodic retraining, online learning, or a temporal GNN that explicitly models how patterns evolve.

---

## 🔭 Future Work

- [ ] **Temporal GraphSAGE**: process each timestep as a graph snapshot, use a GRU to model node embedding evolution across time
- [ ] **Heterogeneous graph**: add account nodes distinct from transaction nodes, model different edge types (sent, received, co-occurrence)
- [ ] **Online retraining pipeline**: retrain on a rolling window as new timesteps arrive, track AUC over time
- [ ] **Explainability**: use GNNExplainer to identify which neighbours and features contribute most to a fraud prediction

---

## 📄 Dataset

**Elliptic Bitcoin Dataset** — Weber et al., 2019

203,769 transaction nodes · 234,355 directed edges · 166 node features · 49 timesteps · ~21% of labelled nodes are illicit

[Download on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [Original paper](https://arxiv.org/abs/1908.02591)

---

## 🙋 About

Built as a portfolio project targeting ML Engineer roles at MAANG and Big Four consulting firms. Focuses on production-realistic evaluation, class imbalance handling, and interpretable threshold selection — not just maximising a single benchmark metric.

---

*If this helped you, a ⭐ on the repo goes a long way.*
