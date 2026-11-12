## Experiment 2: within-dataset test macro-F1 (multiclass)

| Method | UNSW-NB15 | ToN-IoT | BoT-IoT | CIC-IDS2018 | Edge-IIoTset |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.671 ± 0.003** | **0.873 ± 0.001** | **0.989 ± 0.000** | 0.945 ± 0.002 | 0.905 ± 0.000 |
| Small BERT E1 PPFLE | 0.449 ± 0.009 | 0.854 ± 0.001 | 0.935 ± 0.018 | 0.764 ± 0.134 | 0.835 ± 0.004 |
| Small BERT E2 PPFLE-bins | 0.504 ± 0.024 | 0.838 ± 0.003 | 0.955 ± 0.004 | 0.900 ± 0.034 | 0.742 ± 0.003 |
| Small BERT E3 key=value | 0.471 ± 0.019 | 0.856 ± 0.002 | 0.938 ± 0.012 | 0.844 ± 0.136 | 0.817 ± 0.022 |
| Small BERT E4 sentence | 0.411 ± 0.072 | 0.828 ± 0.024 | 0.947 ± 0.004 | 0.768 ± 0.031 | 0.867 ± 0.005 |
| DistilBERT E1 PPFLE | 0.479 ± 0.002 | 0.854 ± 0.002 | 0.896 ± 0.009 | 0.938 ± 0.005 | 0.836 ± 0.003 |
| DistilBERT E2 PPFLE-bins | 0.553 ± 0.003 | 0.846 ± 0.001 | 0.961 ± 0.000 | 0.936 ± 0.003 | 0.746 ± 0.003 |
| DistilBERT E3 key=value | 0.591 ± 0.011 | 0.864 ± 0.002 | 0.979 ± 0.001 | **0.947 ± 0.003** | **0.907 ± 0.001** |
| DistilBERT E4 sentence | 0.522 ± 0.050 | 0.863 ± 0.003 | 0.978 ± 0.002 | 0.944 ± 0.002 | 0.902 ± 0.003 |

Mean ± std over seeds; † = 1 seed (no std); bold = best per column; 'pending' = no results yet. Seeds per cell: XGBoost 3, Small BERT 3, DistilBERT 1/3.
