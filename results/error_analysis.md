# Error analysis (multiclass, within dataset)

From `results/run_details/`: per-class F1 is the mean over seeds; confusion counts are summed over seeds. Confusion matrices: XGBoost and the best encoding (by mean test macro-F1) of each text model. Hard class = even the best method stays below per-class F1 0.5.

## UNSW-NB15

Per-class F1:

| Method | analysis | backdoor | benign | dos | exploits | fuzzers | generic | reconnaissance | shellcode | worms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.833 | 0.535 | 1.000 | 0.679 | 0.505 | 0.557 | 0.391 | 0.516 | 0.791 | 0.900 |
| Small BERT E1 PPFLE | 0.678 | 0.283 | 1.000 | 0.530 | 0.401 | 0.442 | 0.000 | 0.399 | 0.752 | 0.000 |
| Small BERT E2 PPFLE-bins | 0.770 | 0.357 | 1.000 | 0.581 | 0.444 | 0.484 | 0.000 | 0.483 | 0.777 | 0.143 |
| Small BERT E3 key=value | 0.742 | 0.326 | 1.000 | 0.549 | 0.432 | 0.465 | 0.000 | 0.408 | 0.783 | 0.000 |
| Small BERT E4 sentence | 0.573 | 0.176 | 1.000 | 0.524 | 0.408 | 0.405 | 0.000 | 0.262 | 0.763 | 0.000 |
| DistilBERT E1 PPFLE | 0.597 | 0.349 | 1.000 | 0.502 | 0.435 | 0.510 | 0.130 | 0.491 | 0.776 | 0.000 |
| DistilBERT E2 PPFLE-bins | 0.734 | 0.374 | 1.000 | 0.554 | 0.451 | 0.512 | 0.090 | 0.488 | 0.752 | 0.576 |
| DistilBERT E3 key=value | 0.823 | 0.407 | 1.000 | 0.616 | 0.460 | 0.496 | 0.088 | 0.497 | 0.809 | 0.709 |
| DistilBERT E4 sentence | 0.784 | 0.381 | 1.000 | 0.589 | 0.453 | 0.504 | 0.009 | 0.499 | 0.790 | 0.212 |

Hard classes: generic (best F1 0.391)

Top mix-ups, XGBoost (0.671 ± 0.003):

- reconnaissance → fuzzers: 20.7% of reconnaissance (88 rows)
- exploits → dos: 20.1% of exploits (288 rows)
- backdoor → exploits: 16.8% of backdoor (271 rows)
- fuzzers → backdoor: 16.6% of fuzzers (167 rows)
- generic → backdoor: 16.0% of generic (34 rows)

Top mix-ups, Small BERT E2 PPFLE-bins (0.504 ± 0.024):

- worms → dos: 65.1% of worms (41 rows)
- generic → dos: 49.8% of generic (106 rows)
- exploits → dos: 34.2% of exploits (489 rows)
- fuzzers → backdoor: 26.5% of fuzzers (266 rows)
- reconnaissance → dos: 22.3% of reconnaissance (95 rows)

Top mix-ups, DistilBERT E3 key=value (0.591 ± 0.011):

- generic → dos: 40.8% of generic (87 rows)
- exploits → dos: 31.6% of exploits (452 rows)
- worms → dos: 27.0% of worms (17 rows)
- fuzzers → backdoor: 23.2% of fuzzers (233 rows)
- backdoor → exploits: 21.5% of backdoor (347 rows)

## ToN-IoT

Per-class F1:

| Method | backdoor | benign | ddos | dos | injection | mitm | password | ransomware | scanning | xss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.997 | 0.955 | 0.984 | 0.874 | 0.967 | 0.966 | 0.963 | 0.999 | 0.070 | 0.961 |
| Small BERT E1 PPFLE | 0.997 | 0.954 | 0.965 | 0.878 | 0.891 | 0.947 | 0.962 | 0.988 | 0.007 | 0.951 |
| Small BERT E2 PPFLE-bins | 0.990 | 0.953 | 0.959 | 0.760 | 0.880 | 0.934 | 0.961 | 0.989 | 0.019 | 0.934 |
| Small BERT E3 key=value | 0.998 | 0.953 | 0.961 | 0.870 | 0.905 | 0.947 | 0.962 | 0.991 | 0.023 | 0.949 |
| Small BERT E4 sentence | 0.991 | 0.953 | 0.939 | 0.744 | 0.855 | 0.935 | 0.949 | 0.964 | 0.000 | 0.946 |
| DistilBERT E1 PPFLE | 0.994 | 0.954 | 0.956 | 0.876 | 0.894 | 0.939 | 0.962 | 0.999 | 0.013 | 0.949 |
| DistilBERT E2 PPFLE-bins | 0.997 | 0.953 | 0.971 | 0.784 | 0.916 | 0.932 | 0.963 | 0.995 | 0.010 | 0.938 |
| DistilBERT E3 key=value | 0.998 | 0.955 | 0.982 | 0.878 | 0.931 | 0.958 | 0.964 | 0.995 | 0.018 | 0.957 |
| DistilBERT E4 sentence | 0.997 | 0.955 | 0.983 | 0.876 | 0.938 | 0.958 | 0.964 | 0.993 | 0.007 | 0.958 |

Hard classes: scanning (best F1 0.070)

Top mix-ups, XGBoost (0.873 ± 0.001):

- scanning → benign: 96.3% of scanning (1646 rows)
- dos → xss: 14.7% of dos (72 rows)
- dos → mitm: 3.5% of dos (17 rows)
- dos → benign: 3.3% of dos (16 rows)
- password → benign: 3.1% of password (98 rows)

Top mix-ups, Small BERT E3 key=value (0.856 ± 0.002):

- scanning → benign: 98.7% of scanning (1687 rows)
- dos → xss: 14.7% of dos (72 rows)
- injection → xss: 4.9% of injection (54 rows)
- dos → mitm: 4.3% of dos (21 rows)
- injection → ddos: 2.7% of injection (30 rows)

Top mix-ups, DistilBERT E3 key=value (0.864 ± 0.002):

- scanning → benign: 98.9% of scanning (1691 rows)
- dos → xss: 14.7% of dos (72 rows)
- injection → xss: 4.5% of injection (50 rows)
- dos → mitm: 3.5% of dos (17 rows)
- dos → benign: 2.0% of dos (10 rows)

## BoT-IoT

Per-class F1:

| Method | benign | ddos | dos | reconnaissance | theft |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.985 | 0.994 | 0.994 | 0.971 | 1.000 |
| Small BERT E1 PPFLE | 0.969 | 0.932 | 0.942 | 0.938 | 0.894 |
| Small BERT E2 PPFLE-bins | 0.968 | 0.956 | 0.962 | 0.936 | 0.953 |
| Small BERT E3 key=value | 0.970 | 0.950 | 0.956 | 0.939 | 0.874 |
| Small BERT E4 sentence | 0.970 | 0.944 | 0.951 | 0.942 | 0.930 |
| DistilBERT E1 PPFLE | 0.970 | 0.791 | 0.812 | 0.938 | 0.970 |
| DistilBERT E2 PPFLE-bins | 0.970 | 0.957 | 0.963 | 0.938 | 0.975 |
| DistilBERT E3 key=value | 0.985 | 0.981 | 0.984 | 0.971 | 0.973 |
| DistilBERT E4 sentence | 0.985 | 0.978 | 0.980 | 0.970 | 0.978 |

Hard classes: none

Top mix-ups, XGBoost (0.989 ± 0.000):

- reconnaissance → benign: 2.5% of reconnaissance (122 rows)
- benign → reconnaissance: 1.6% of benign (151 rows)
- ddos → dos: 0.7% of ddos (142 rows)
- dos → ddos: 0.5% of dos (121 rows)
- benign → dos: 0.0% of benign (3 rows)

Top mix-ups, Small BERT E2 PPFLE-bins (0.955 ± 0.004):

- reconnaissance → benign: 10.6% of reconnaissance (510 rows)
- ddos → dos: 5.5% of ddos (1175 rows)
- dos → ddos: 2.8% of dos (676 rows)
- theft → benign: 2.0% of theft (4 rows)
- theft → reconnaissance: 1.0% of theft (2 rows)

Top mix-ups, DistilBERT E3 key=value (0.979 ± 0.001):

- reconnaissance → benign: 3.7% of reconnaissance (178 rows)
- ddos → dos: 2.3% of ddos (499 rows)
- theft → benign: 2.0% of theft (4 rows)
- dos → ddos: 1.2% of dos (297 rows)
- benign → reconnaissance: 1.0% of benign (88 rows)

## CIC-IDS2018

Per-class F1:

| Method | benign | bot | ddos_hoic | ddos_loic_http | ddos_loic_udp | dos_goldeneye | dos_hulk | dos_slowloris | infiltration | ssh_bruteforce | web_attack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.996 | 1.000 | 1.000 | 0.999 | 1.000 | 0.998 | 1.000 | 0.991 | 0.490 | 1.000 | 0.922 |
| Small BERT E1 PPFLE | 0.996 | 0.302 | 0.999 | 1.000 | 0.994 | 0.857 | 0.999 | 0.294 | 0.143 | 1.000 | 0.822 |
| Small BERT E2 PPFLE-bins | 0.996 | 0.887 | 1.000 | 1.000 | 0.999 | 0.983 | 0.988 | 0.932 | 0.272 | 1.000 | 0.841 |
| Small BERT E3 key=value | 0.996 | 0.641 | 0.999 | 1.000 | 0.997 | 0.910 | 0.999 | 0.591 | 0.288 | 1.000 | 0.867 |
| Small BERT E4 sentence | 0.995 | 0.633 | 1.000 | 0.999 | 0.995 | 0.847 | 0.991 | 0.000 | 0.129 | 0.999 | 0.860 |
| DistilBERT E1 PPFLE | 0.996 | 0.961 | 1.000 | 1.000 | 1.000 | 0.992 | 0.998 | 0.973 | 0.475 | 0.999 | 0.925 |
| DistilBERT E2 PPFLE-bins | 0.996 | 0.988 | 1.000 | 0.999 | 1.000 | 0.994 | 0.998 | 0.982 | 0.444 | 1.000 | 0.893 |
| DistilBERT E3 key=value | 0.996 | 0.993 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.494 | 0.999 | 0.934 |
| DistilBERT E4 sentence | 0.996 | 0.993 | 1.000 | 1.000 | 1.000 | 0.998 | 0.999 | 0.982 | 0.489 | 1.000 | 0.931 |

Hard classes: infiltration (best F1 0.494)

Top mix-ups, XGBoost (0.945 ± 0.002):

- infiltration → benign: 62.6% of infiltration (291 rows)
- web_attack → benign: 6.0% of web_attack (31 rows)
- dos_slowloris → dos_goldeneye: 1.8% of dos_slowloris (1 rows)
- infiltration → web_attack: 1.3% of infiltration (6 rows)
- infiltration → ddos_loic_http: 0.6% of infiltration (3 rows)

Top mix-ups, Small BERT E2 PPFLE-bins (0.900 ± 0.034):

- infiltration → benign: 70.3% of infiltration (327 rows)
- infiltration → web_attack: 11.2% of infiltration (52 rows)
- web_attack → benign: 6.2% of web_attack (32 rows)
- web_attack → infiltration: 3.3% of web_attack (17 rows)
- bot → benign: 2.6% of bot (2 rows)

Top mix-ups, DistilBERT E3 key=value (0.947 ± 0.003):

- infiltration → benign: 62.2% of infiltration (289 rows)
- web_attack → benign: 2.3% of web_attack (12 rows)
- infiltration → web_attack: 2.2% of infiltration (10 rows)
- bot → benign: 1.3% of bot (1 rows)
- web_attack → infiltration: 0.6% of web_attack (3 rows)

## Edge-IIoTset

Per-class F1:

| Method | backdoor | benign | ddos_http | ddos_icmp | ddos_tcp | ddos_udp | fingerprinting | mitm | password | port_scanning | ransomware | sql_injection | uploading | vulnerability_scanner | xss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.970 | 0.982 | 0.764 | 1.000 | 1.000 | 1.000 | 0.834 | 1.000 | 0.838 | 0.886 | 0.935 | 0.802 | 0.756 | 0.964 | 0.850 |
| Small BERT E1 PPFLE | 0.666 | 0.980 | 0.608 | 0.998 | 1.000 | 0.997 | 0.825 | 0.971 | 0.834 | 0.833 | 0.616 | 0.778 | 0.671 | 0.932 | 0.823 |
| Small BERT E2 PPFLE-bins | 0.835 | 0.981 | 0.636 | 0.992 | 1.000 | 1.000 | 0.715 | 1.000 | 0.559 | 0.873 | 0.584 | 0.618 | 0.249 | 0.902 | 0.183 |
| Small BERT E3 key=value | 0.914 | 0.981 | 0.730 | 0.970 | 1.000 | 0.999 | 0.114 | 0.974 | 0.815 | 0.831 | 0.788 | 0.781 | 0.669 | 0.927 | 0.767 |
| Small BERT E4 sentence | 0.914 | 0.981 | 0.730 | 0.990 | 0.999 | 0.999 | 0.754 | 0.991 | 0.833 | 0.831 | 0.803 | 0.786 | 0.657 | 0.943 | 0.797 |
| DistilBERT E1 PPFLE | 0.596 | 0.980 | 0.614 | 0.999 | 1.000 | 0.999 | 0.840 | 1.000 | 0.833 | 0.833 | 0.614 | 0.777 | 0.672 | 0.943 | 0.840 |
| DistilBERT E2 PPFLE-bins | 0.852 | 0.981 | 0.643 | 0.992 | 1.000 | 1.000 | 0.715 | 0.994 | 0.574 | 0.877 | 0.624 | 0.619 | 0.264 | 0.900 | 0.156 |
| DistilBERT E3 key=value | 0.972 | 0.982 | 0.769 | 1.000 | 1.000 | 1.000 | 0.852 | 1.000 | 0.839 | 0.877 | 0.935 | 0.810 | 0.749 | 0.966 | 0.855 |
| DistilBERT E4 sentence | 0.970 | 0.982 | 0.770 | 1.000 | 1.000 | 1.000 | 0.852 | 0.997 | 0.839 | 0.848 | 0.934 | 0.811 | 0.715 | 0.965 | 0.851 |

Hard classes: none

Top mix-ups, XGBoost (0.905 ± 0.000):

- ddos_http → benign: 30.1% of ddos_http (579 rows)
- uploading → benign: 29.8% of uploading (435 rows)
- password → benign: 27.4% of password (543 rows)
- sql_injection → benign: 25.5% of sql_injection (516 rows)
- xss → benign: 23.8% of xss (144 rows)

Top mix-ups, Small BERT E4 sentence (0.867 ± 0.005):

- uploading → benign: 30.0% of uploading (439 rows)
- ddos_http → benign: 30.0% of ddos_http (577 rows)
- password → benign: 27.5% of password (545 rows)
- sql_injection → benign: 25.5% of sql_injection (516 rows)
- xss → benign: 24.1% of xss (146 rows)

Top mix-ups, DistilBERT E3 key=value (0.907 ± 0.001):

- ddos_http → benign: 30.0% of ddos_http (577 rows)
- uploading → benign: 29.8% of uploading (435 rows)
- password → benign: 27.4% of password (544 rows)
- sql_injection → benign: 25.5% of sql_injection (516 rows)
- xss → benign: 23.8% of xss (144 rows)

Figures: `report/generated/figures/perclass_f1_nf_unsw_nb15.*`, `report/generated/figures/confusion_nf_unsw_nb15.*`, `report/generated/figures/perclass_f1_nf_ton_iot.*`, `report/generated/figures/confusion_nf_ton_iot.*`, `report/generated/figures/perclass_f1_nf_bot_iot.*`, `report/generated/figures/confusion_nf_bot_iot.*`, `report/generated/figures/perclass_f1_nf_cse_cic_ids2018.*`, `report/generated/figures/confusion_nf_cse_cic_ids2018.*`, `report/generated/figures/perclass_f1_edge_iiotset.*`, `report/generated/figures/confusion_edge_iiotset.*`
