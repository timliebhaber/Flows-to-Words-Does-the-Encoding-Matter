## Experiment 1: replication on Edge-IIoTset

| Method | fixed ("0"/"0.0" unified): accuracy | fixed ("0"/"0.0" unified): macro-F1 | as published: accuracy | as published: macro-F1 |
|---|---:|---:|---:|---:|
| XGBoost (multiclass) | 0.967 ± 0.000 | 0.905 ± 0.000 | 0.985 ± 0.000 | 0.931 ± 0.001 |
| XGBoost (binary) | 0.973 ± 0.000 | 0.965 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Small BERT E1 PPFLE (multiclass) | 0.955 ± 0.000 | 0.835 ± 0.004 | 0.978 ± 0.003 | 0.863 ± 0.031 |
| Small BERT E2 PPFLE-bins (multiclass) | 0.941 ± 0.001 | 0.742 ± 0.003 | -- | -- |
| Small BERT E3 key=value (multiclass) | 0.958 ± 0.002 | 0.817 ± 0.022 | -- | -- |
| Small BERT E4 sentence (multiclass) | 0.961 ± 0.001 | 0.867 ± 0.005 | -- | -- |
| DistilBERT E1 PPFLE (multiclass) | 0.955 ± 0.001 | 0.836 ± 0.003 | -- | -- |
| DistilBERT E2 PPFLE-bins (multiclass) | 0.942 ± 0.001 | 0.746 ± 0.003 | -- | -- |
| DistilBERT E3 key=value (multiclass) | 0.967 ± 0.000 | 0.907 ± 0.001 | -- | -- |
| DistilBERT E4 sentence (multiclass) | 0.967 ± 0.000 | 0.902 ± 0.003 | -- | -- |
| SecurityBERT paper (E1 PPFLE) |  |  | 0.982 |  |

Mean ± std over seeds; † = 1 seed (no std); bold = best per column; 'pending' = no results yet. Seeds per cell: XGBoost 3, Small BERT 3, DistilBERT 3.
