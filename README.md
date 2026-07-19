\# Graph-Based Fraud Detection using GCN \& GAT



A comparison of graph neural networks (GCN, GAT) against a traditional XGBoost baseline for e-commerce fraud detection, built to explore whether modeling transactions as a connected graph improves detection of coordinated fraud.



\---



\## Table of Contents

\- \[Objective](#objective)

\- \[Dataset](#dataset)

\- \[Project Structure](#project-structure)

\- \[Setup \& Installation](#setup--installation)

\- \[Methodology](#methodology)

\- \[Key Findings](#key-findings)

\- \[Results Summary](#results-summary)

\- \[Conclusion](#conclusion)

\- \[Key Code Reference](#key-code-reference)

\- \[Future Work](#future-work)



\---



\## Objective



Traditional fraud detection models treat every transaction independently, as an isolated row of features. Real-world fraud is often \*\*coordinated\*\* — multiple transactions linked through a shared device or IP address. This project builds a \*\*Graph Neural Network\*\* that explicitly models these relationships as graph structure, and tests whether that structural awareness improves fraud detection compared to a strong traditional baseline (XGBoost).



This project was built to go beyond prior tabular/text classification work (house price prediction, SMS spam detection, telco churn) into a new paradigm — graph-based machine learning — while covering a broad span of core ML/DL concepts: classical ML, class imbalance handling, graph construction, GNN architectures (GCN and GAT), and rigorous model evaluation and diagnosis.



\---



\## Dataset



\*\*Source:\*\* \[Fraud\_Ecommerce dataset on Kaggle](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)



\- \*\*151,112 transactions\*\*, 11 original columns, no missing values

\- \*\*Target:\*\* `class` (1 = fraud, 0 = legitimate) — 9.36% fraud rate (14,151 fraud / 136,961 legitimate)

\- \*\*Key columns:\*\* `user\_id`, `signup\_time`, `purchase\_time`, `purchase\_value`, `device\_id`, `source`, `browser`, `sex`, `age`, `ip\_address`, `class`

\- This dataset was chosen specifically because it includes explicit `device\_id` and `ip\_address` fields — needed to construct a meaningful graph — and has a more balanced fraud rate than typical fraud datasets (many sit around 3–4%), giving models enough real fraud examples to learn from.



\*\*Setup:\*\* download both `Fraud\_Data.csv` and `IpAddress\_to\_Country.csv` from the Kaggle link above and place them in `data/raw/` (this folder is not tracked in Git — see \[Setup \& Installation](#setup--installation)).



\---



\## Project Structure



```

gnn-fraud-detection/

│

├── data/

│   ├── raw/                  ← Fraud\_Data.csv, IpAddress\_to\_Country.csv (not tracked in Git)

│   └── processed/            ← cleaned/processed data

│

├── notebooks/

│   └── 01\_eda\_and\_baseline.ipynb   ← full pipeline: EDA → XGBoost → graph → GCN → GAT

│

├── outputs/

│   └── figures/               ← saved plots (e.g., fraud\_rate\_by\_sharing.png)

│

├── README.md

└── requirements.txt

```



\---



\## Setup \& Installation



\### Requirements

\- Python 3.10+ (tested on 3.10 and 3.14)

\- pip or conda



\### Install dependencies

```bash

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install torch\_geometric

pip install pandas numpy scikit-learn matplotlib networkx xgboost jupyter jupyterlab

```



Or, if `requirements.txt` is present:

```bash

pip install -r requirements.txt

```



\### Get the dataset

1\. Download from \[Kaggle: Fraud\_Ecommerce](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)

2\. Place `Fraud\_Data.csv` and `IpAddress\_to\_Country.csv` into `data/raw/`



\### Run

```bash

jupyter lab

```

Open `notebooks/01\_eda\_and\_baseline.ipynb` and run all cells top to bottom.



\*\*Note:\*\* this project is CPU-only — no GPU required. Dataset size was deliberately kept moderate (\~151k rows) to keep graph construction and training fast and laptop-friendly.



\---



\## Methodology



The notebook follows this sequence — each step's purpose is noted for quick recall:



| Step | What it does | Why |

|---|---|---|

| 1. Load \& inspect data | Load CSV, check shape, preview | Confirm data loaded correctly before any processing |

| 2. Class balance check | Count fraud vs. legitimate | Establishes the imbalance problem to design around |

| 3. Missing values \& shared-ID check | Check nulls; count shared device\_ids/IPs | Confirms data is clean and that enough sharing exists to build a meaningful graph |

| 4. Hypothesis validation | Fraud rate when device/IP is shared vs. not | Validates the core project premise \*before\* building any model |

| 5. Visualization | Bar chart of fraud rate by sharing | Produces a reusable presentation asset |

| 6. Feature engineering | `time\_since\_signup`, one-hot encoding | Prepares features for modeling |

| 7. Train/test split | Stratified 80/20 split | Preserves fraud ratio in both sets |

| 8. XGBoost baseline | Trained with `scale\_pos\_weight` | Establishes a strong, relationship-unaware baseline |

| 9. Graph construction | Nodes = transactions, edges = shared device/IP | Encodes the coordinated-fraud hypothesis into structure |

| 10. PyG conversion | Build `Data` object, train/test masks | Prepares graph for PyTorch Geometric |

| 11–13. GCN model | Build, train, evaluate | Tests whether neighbor-averaging improves detection |

| 14. Feature scaling fix | StandardScaler on node features | Fixes initial GCN underperformance (unscaled features) |

| 15. Connected vs. isolated analysis | Split test set by node degree | Fairer test of the graph hypothesis than an overall metric |

| 16–17. GAT model | Build, train, evaluate | Tests whether attention (vs. averaging) closes the gap |



\---



\## Key Findings



\### 1. The core hypothesis was strongly validated

Fraud rate when a device is shared: \*\*52.5%\*\* vs. \*\*3.0%\*\* when not shared.

Fraud rate when an IP is shared: \*\*91.3%\*\* vs. \*\*4.6%\*\* when not shared.

→ Coordinated fraud via shared infrastructure is a real, strong pattern in this data.



\### 2. Graph structure alone (91% of nodes had no neighbors) limits opportunity

\- 151,112 nodes, 51,708 edges, \*\*87% of nodes isolated\*\* (no shared device/IP with anything).

\- This is expected given how rare sharing was found to be (6,175 shared devices, 760 shared IPs out of 151k transactions).

\- Implication: GNN structural advantage can only apply to the \~13% connected subset.



\### 3. First GCN attempt underperformed — diagnosed as a feature-scaling issue

\- Initial GCN: PR-AUC \*\*0.40\*\* (worse than XGBoost's 0.71).

\- Cause: unscaled features (`purchase\_value`, `time\_since\_signup` had far larger ranges than boolean flags), which destabilizes neural network training (tree models like XGBoost are unaffected by this).

\- Fix: `StandardScaler` on node features → GCN PR-AUC improved to \*\*0.66\*\*.



\### 4. XGBoost outperforms both GCN and GAT — especially on connected nodes

| | Connected nodes (n=3,891) | Isolated nodes (n=26,332) |

|---|---|---|

| XGBoost | \*\*0.93\*\* | 0.03 |

| GCN | 0.52 | 0.03 |

| GAT | 0.52 | 0.03 |



\### 5. Switching to attention (GAT) did not close the gap — ruling out oversmoothing

\- Hypothesis: GCN's simple neighbor-averaging "blurs" fraud/legitimate signals in mixed clusters (oversmoothing); GAT's learned attention weights should fix this.

\- Result: GAT scored virtually identically to GCN (0.524 vs. 0.519) on connected nodes — no meaningful improvement.

\- \*\*Real explanation:\*\* the `device\_shared`/`ip\_shared` flags are already a clean, explicit, highly predictive feature. XGBoost (tree-based) can exploit this with a single split. Neural networks (GCN or GAT) must \*learn\* an approximation of that same logic through weighted sums — a harder, noisier path to the same insight the flag already provides directly.



\---



\## Results Summary



| Metric | XGBoost | GCN | GAT |

|---|---|---|---|

| Overall PR-AUC | \*\*0.71\*\* | 0.66 | 0.66 |

| Fraud recall | 0.71 | 0.71 | 0.71 |

| Fraud precision | 0.54 | 0.52 | 0.52 |

| PR-AUC (connected nodes) | \*\*0.93\*\* | 0.52 | 0.52 |

| PR-AUC (isolated nodes) | 0.03 | 0.03 | 0.03 |



\---



\## Conclusion



> When a strong relational pattern (shared device/IP) is already available as an explicit engineered feature, a tree-based model can exploit it directly and efficiently. A graph neural network — whether using simple averaging (GCN) or learned attention (GAT) — has to \*learn\* an approximation of that same logic, which is a harder task that didn't pay off here. Graph-based methods would likely show a clearer advantage on datasets where the relational signal is \*not\* already flattened into a feature, or on tasks requiring multi-hop reasoning that a single flag can't express.



This is a legitimate, evidence-based negative result — arrived at by testing a specific hypothesis (oversmoothing) directly with a second architecture, rather than assuming an explanation. Understanding \*why\* a more sophisticated model doesn't win is a valuable, demonstrable skill in its own right.



\---



\## Key Code Reference



Quick lookup for methods used, in case a report needs to reference "how" something was done:



\- \*\*`value\_counts()` / `.map()`\*\* — counting occurrences and looking up per-row counts (used to flag shared devices/IPs)

\- \*\*Boolean filtering (`df\[df\['col']]`)\*\* — filtering rows by a True/False condition, used throughout to compute group-specific fraud rates

\- \*\*`pd.get\_dummies(..., drop\_first=True)`\*\* — one-hot encoding categorical columns for model input

\- \*\*`train\_test\_split(..., stratify=y)`\*\* — splitting data while preserving class ratio

\- \*\*`scale\_pos\_weight`\*\* — XGBoost's built-in class imbalance correction

\- \*\*`precision\_recall\_curve` + `auc`\*\* — PR-AUC, the primary evaluation metric (more robust than accuracy on imbalanced data)

\- \*\*`networkx.Graph()` + `combinations()`\*\* — building the transaction graph from shared attributes

\- \*\*PyTorch Geometric `Data` object\*\* — bundles node features (`x`), graph connectivity (`edge\_index`), and labels (`y`)

\- \*\*Train/test masks\*\* — boolean arrays marking which nodes belong to train/test, used instead of physically splitting the graph (since GNNs need the full graph intact)

\- \*\*`GCNConv` / `GATConv`\*\* — the graph convolution/attention layers from PyTorch Geometric; both follow the same `\_\_init\_\_`/`forward` class pattern as any PyTorch model

\- \*\*`StandardScaler`\*\* — feature scaling, required for stable neural network training (not required for XGBoost)

\- \*\*Manual training loop\*\* (`zero\_grad → forward → loss → backward → step`) — the explicit version of what `.fit()` does automatically in scikit-learn/XGBoost



\*(For line-by-line explanations of every code cell, see the full presentation script / notebook markdown cells.)\*



\---



\## Future Work



\- Engineer a "neighbor fraud rate" or "cluster size" feature and feed it directly to XGBoost, to test whether it alone matches the GNN's ceiling.

\- Test this approach on a dataset where the relational signal is \*not\* already flattened into an explicit feature, to see if GNN advantage appears more clearly.

\- Try deeper GNN architectures or different aggregation strategies (e.g., GraphSAGE) as an additional comparison point.

\- Explore multi-hop relationships (e.g., transactions connected through a chain of shared devices) rather than only direct 1-hop edges.

