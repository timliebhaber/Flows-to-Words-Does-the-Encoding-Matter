## Experiment 2: within-dataset test accuracy (multiclass)

| Method | UNSW-NB15 | ToN-IoT | BoT-IoT | CIC-IDS2018 | Edge-IIoTset |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.945 ± 0.000** | **0.941 ± 0.000** | **0.991 ± 0.000** | 0.993 ± 0.000 | 0.967 ± 0.000 |
| Small BERT E1 PPFLE | 0.925 ± 0.001 | 0.935 ± 0.001 | 0.942 ± 0.002 | 0.989 ± 0.003 | 0.955 ± 0.000 |
| Small BERT E2 PPFLE-bins | 0.933 ± 0.000 | 0.930 ± 0.001 | 0.959 ± 0.001 | 0.991 ± 0.001 | 0.941 ± 0.001 |
| Small BERT E3 key=value | 0.930 ± 0.002 | 0.935 ± 0.001 | 0.954 ± 0.002 | 0.991 ± 0.002 | 0.958 ± 0.002 |
| Small BERT E4 sentence | 0.923 ± 0.006 | 0.928 ± 0.003 | 0.951 ± 0.004 | 0.989 ± 0.001 | 0.961 ± 0.001 |
| DistilBERT E1 PPFLE | 0.928 ± 0.001 | 0.934 ± 0.000 | 0.840 ± 0.022 | 0.993 ± 0.000 | 0.955 ± 0.001 |
| DistilBERT E2 PPFLE-bins | 0.932 ± 0.001 | 0.932 ± 0.001 | 0.960 ± 0.002 | 0.992 ± 0.000 | 0.942 ± 0.001 |
| DistilBERT E3 key=value | 0.937 ± 0.001 | 0.939 ± 0.001 | 0.982 ± 0.003 | **0.993 ± 0.000** | **0.967 ± 0.000** |
| DistilBERT E4 sentence | 0.934 ± 0.003 | 0.939 ± 0.001 | 0.979 ± 0.002 | 0.993 ± 0.000 | 0.967 ± 0.000 |

Mean ± std over seeds; † = 1 seed (no std); bold = best per column; 'pending' = no results yet. Seeds per cell: XGBoost 3, Small BERT 3, DistilBERT 1/3.
