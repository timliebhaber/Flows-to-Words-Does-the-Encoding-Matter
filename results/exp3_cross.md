## Experiment 3: cross-dataset transfer (binary), mean over the train x test grid

| Method | macro-F1 same dataset | macro-F1 off-diagonal | attack recall same dataset | attack recall off-diagonal | FPR same dataset | FPR off-diagonal |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.979 ± 0.000 | 0.552 ± 0.018 | **0.966 ± 0.000** | 0.575 ± 0.019 | 0.009 ± 0.000 | 0.347 ± 0.033 |
| Small BERT E1 PPFLE | 0.977 ± 0.000 | 0.424 ± 0.008 | 0.961 ± 0.001 | 0.478 ± 0.043 | 0.009 ± 0.001 | 0.481 ± 0.033 |
| Small BERT E2 PPFLE-bins | 0.976 ± 0.000 | 0.524 ± 0.010 | 0.962 ± 0.002 | 0.559 ± 0.012 | 0.010 ± 0.000 | 0.411 ± 0.017 |
| Small BERT E3 key=value | 0.978 ± 0.000 | 0.496 ± 0.002 | 0.964 ± 0.001 | 0.464 ± 0.033 | 0.011 ± 0.002 | 0.319 ± 0.065 |
| Small BERT E4 sentence | 0.978 ± 0.000 | 0.486 ± 0.015 | 0.964 ± 0.001 | 0.461 ± 0.034 | 0.011 ± 0.002 | 0.347 ± 0.027 |
| DistilBERT E1 PPFLE | 0.977 ± 0.000 | 0.517 ± 0.023 | 0.963 ± 0.000 | 0.470 ± 0.035 | 0.008 ± 0.000 | 0.290 ± 0.008 |
| DistilBERT E2 PPFLE-bins | 0.977 ± 0.000 | **0.594 ± 0.025** | 0.963 ± 0.000 | **0.642 ± 0.042** | 0.008 ± 0.000 | 0.363 ± 0.017 |
| DistilBERT E3 key=value | 0.980 ± 0.000 | 0.485 ± 0.006 | 0.965 ± 0.000 | 0.383 ± 0.064 | 0.008 ± 0.000 | **0.262 ± 0.078** |
| DistilBERT E4 sentence | **0.981 ± 0.000** | 0.479 ± 0.007 | 0.965 ± 0.000 | 0.407 ± 0.067 | **0.008 ± 0.000** | 0.275 ± 0.056 |

Off-diagonal = trained on one dataset, tested on another (the generalisation number). Per seed, the 4 diagonal and 12 off-diagonal cells are averaged first. Mean ± std over seeds; † = 1 seed (no std); bold = best per column; 'pending' = no results yet. Seeds per cell: XGBoost 3, Small BERT 3, DistilBERT 3.
