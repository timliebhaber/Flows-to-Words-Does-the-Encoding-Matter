# Report outline: "Flows to Words: Does the Encoding Matter?"

What each section of the report (`report/sections/*.tex`, put together by `report/main.tex`) says and which generated number, table or figure backs it.

**Rule: no number is typed by hand.** Every value is a key, written here as `\res{<key>}`. You copy that same
`\res{<key>}` into the LaTeX, and it prints the current value.
- All keys and their current values: `results/numbers_analysis.md` and `results/numbers_privacy.md`.
- To regenerate: `uv run ftw analysis privacy`.
- `\resdiff{<a>}{<b>}` prints a − b; `\respct{<a>}{<b>}` prints 100·a/b.
- After each run, the analysis warns about any key used here or in the report's `.tex` files that has no value yet.
- Statements marked *(check `hyp.…`)* depend on the results: re-read them when that verdict changes.

Key patterns (m = model, e = encoding, d = dataset):
- `exp2.<m>.<e>.<d>.macro_f1`: within-dataset test macro-F1, mean ± std; `.mean`, `.std`, `.n` give the parts.
- `exp3.<m>.<e>.<off|diag>.<metric>`: cross-dataset averages; `exp3.<m>.<e>.<train d>.<test d>.<metric>` for one
  cell.
- `exp1.<m>.<e>.<edge_iiotset|edge_iiotset_raw>.<task>.<metric>`.
- `hyp.<RQ1|H1..H4>.verdict`, `.held` and `.total`.
- `privacy.<e>.<level>.<token_share|row_share>.<min|max>`, and `privacy.<d>.<e>.<level>.<share>` per dataset.
- Settings: `cfg.<config file>.<path>`. Stats files: `data.*`, `leakage.*`, `tokenizer.*`, `encoding.*`,
  `tuning.*`.

## 1. Introduction

- Problem: NIDS on flow tables; recent work turns each row into text for language models. SecurityBERT/PPFLE hashes
  every `column$value` and reports `\res{exp1.paper.accuracy}` accuracy on Edge-IIoTset.
- Gap: the encoding choice is never isolated, the baselines were weak, and cross-dataset transfer and the privacy
  claim are untested.
- This work: every encoding (E1–E4) with each language model (small BERT, DistilBERT) and a tuned XGBoost, on the
  NetFlow datasets and Edge-IIoTset, within and across datasets, plus a replication and a privacy attack.
- RQ1–RQ4 and H1–H4 (plan §2), stated as hypotheses to test.
- Contributions:
  1. a controlled encoding benchmark;
  2. the Edge-IIoTset label leak;
  3. a dictionary attack on PPFLE;
  4. when NLP helps vs. plain XGBoost.

## 2. Related work

- SecurityBERT / PPFLE; Edge-IIoTset.
- NetFlow standard feature set and the UQ NIDS datasets.
- "From Flows to Words" (prompt-only LLMs on flows).
- Serialising tables as text for language models (cite one or two general works); DistilBERT; XGBoost.

## 3. Data

- **NetFlow v3**: four datasets with an identical column set, which is what makes cross-dataset tests possible.
  Features kept: `\res{cfg.data.columns.features.count}`.
- **Leaky identifiers dropped** (`\res{cfg.data.columns.leaky.count}` columns: IPs, `L4_SRC_PORT`, timestamps,
  `DNS_QUERY_ID`). `L4_DST_PORT` is kept.
- **Sampling**: `\res{cfg.data.sampling.sample_size}` rows per dataset, stratified by class. Classes under
  `\res{cfg.data.sampling.rare_class_threshold}` rows are kept whole.
- **NF-BoT-IoT** keeps all benign rows, because the dataset is almost all attack:
  - benign share in the sample: `\respct{data.nf_bot_iot.binary_counts.train.0}{data.nf_bot_iot.rows.train}` %
    of train;
  - raw rows: `\res{data.nf_bot_iot.rows.raw}`.
- **Dedupe after dropping columns** removes `\respct{data.<d>.rows.exact_duplicates_removed}{data.<d>.rows.sampled}`
  % of each sample. Rows with the same features but a different label are dropped. Then a stratified split
  `\res{cfg.data.split.train}` / `\res{cfg.data.split.val}` / `\res{cfg.data.split.test}`.
- **Table**: per-dataset rows raw → sampled → final → train/val/test, built from `data.<d>.rows.*`
  (`results/dataset_stats.md` has the same as Markdown).
- **Edge-IIoTset**:
  - raw rows: `\res{data.edge_iiotset.rows.raw}`;
  - dropped columns: `\res{cfg.data_edge.columns.leaky.count}` (the authors' recipe);
  - features left: `\res{cfg.data_edge.columns.features.count}`.
- **Leakage check**: XGBoost on one feature at a time, train → val.
  - UNSW-NB15: `MIN_TTL` alone reaches binary macro-F1 `\res{leakage.nf_unsw_nb15.binary.MIN_TTL}`, with
    `MAX_TTL` at `\res{leakage.nf_unsw_nb15.binary.MAX_TTL}`. Attack and benign traffic come from different hosts
    (TTL fingerprint).
  - In multiclass the same feature only reaches `\res{leakage.nf_unsw_nb15.multiclass.MIN_TTL}`.
  - Decision: all features kept, and the issue documented.

## 4. Finding: a label leak in Edge-IIoTset

- Empty fields are written `"0"` in the normal-traffic files and `"0.0"` in the attack files (also reported in
  arXiv 2608.15761).
- Evidence, single-feature binary macro-F1 as published vs fixed:
  - `\res{leakage.edge_iiotset_raw.binary.mqtt.topic}` vs `\res{leakage.edge_iiotset.binary.mqtt.topic}`;
  - the same for `mqtt.protoname`, `mqtt.conack.flags` and `dns.qry.name.len`.
- Effect on XGBoost:
  - accuracy as published `\res{exp1.xgboost.none.edge_iiotset_raw.multiclass.accuracy}` vs fixed
    `\res{exp1.xgboost.none.edge_iiotset.multiclass.accuracy}`;
  - binary macro-F1 `\res{exp1.xgboost.none.edge_iiotset_raw.binary.macro_f1}` vs
    `\res{exp1.xgboost.none.edge_iiotset.binary.macro_f1}`.
- Effect on our replication, small BERT on E1 PPFLE:
  - accuracy as published `\res{exp1.bert_small.e1_ppfle.edge_iiotset_raw.multiclass.accuracy}` vs fixed
    `\res{exp1.bert_small.e1_ppfle.edge_iiotset.multiclass.accuracy}`;
  - the paper reports `\res{exp1.paper.accuracy}`.
- Interpretation *(check `hyp.RQ1.verdict`)*: the published number matches the leaky version.
- Also: `dns.qry.name` and `dns.qry.name.len` look swapped in the raw file (kept as published).

## 5. Method

- **Encodings** (one example row each from `results/encoding_stats.md`):
  - E1: `\res{cfg.encoders.hash.algorithm}` of `column$value`, `\res{cfg.encoders.hash.length}` hex characters.
  - E2: the same on `\res{cfg.encoders.binning.n_bins}` quantile bins fitted on train.
  - E3: `key=value` text.
  - E4: sentence template.
- **XGBoost**: grid on val, `\res{seeds.xgboost}` seeds.
- **Small BERT**:
  - `\res{cfg.bert_small.model.num_hidden_layers}` layers, hidden `\res{cfg.bert_small.model.hidden_size}`,
    `\res{cfg.bert_small.model.num_attention_heads}` heads, FFN `\res{cfg.bert_small.model.intermediate_size}`,
    which gives `\res{params.bert_small.millions}` M parameters. The paper states `\res{exp1.paper.params_millions}` M;
    say that we kept the stated sizes.
  - Byte-level BPE, vocab `\res{cfg.bert_small.tokenizer.vocab_size}`, per encoding on train. max_len covers the
    `\res{cfg.bert_small.tokenizer.coverage_percentile}`th percentile, e.g. `\res{tokenizer.e1_ppfle.nf_unsw_nb15.max_len}`.
  - `\res{cfg.bert_small.training.epochs}` epochs, `\res{seeds.bert_small}` seeds.
- **DistilBERT**:
  - `\res{params.distilbert.millions}` M parameters, `\res{cfg.distilbert.training.epochs}` epochs, batch
    `\res{cfg.distilbert.training.batch_size}`, `\res{seeds.distilbert}` seed.
  - E1 needs `\res{encoding.stats.nf_unsw_nb15.encodings.e1_ppfle.wordpiece_tokens.p99}` WordPieces (p99), so E1/E2
    are cut at `\res{cfg.distilbert.tokenizer.max_len_cap}`.
  - Side runs with shorter hashes (nothing cut):
    - E1: `\res{exp2.distilbert.e1_ppfle_h8.nf_unsw_nb15.macro_f1}` vs `\res{exp2.distilbert.e1_ppfle.nf_unsw_nb15.macro_f1}`;
    - E2: `\res{exp2.distilbert.e2_ppfle_bins_h8.nf_unsw_nb15.macro_f1}` vs `\res{exp2.distilbert.e2_ppfle_bins.nf_unsw_nb15.macro_f1}`.
- **Tuning**: pilot LR grid on one dataset (`\res{cfg.distilbert.tuning.pilot.dataset}`); winners per encoding in
  `tuning.<m>.<e>.<d>.multiclass.best.lr`.
- **Verdict rule**: two numbers differ only if |Δ| > max(`\res{cfg.analysis.hypotheses.tie_margin}`, the larger
  seed std). "Does well" = within `\res{cfg.analysis.hypotheses.well_margin}` of the best method.

## 6. Results

### 6.1 Exp 1: replication (RQ1)
- Table `generated/exp1_replication.tex`; numbers as in §4.
- Verdict: `\res{hyp.RQ1.verdict}`.

### 6.2 Exp 2: within-dataset (RQ2, H1, H2)
- Table `generated/exp2_within_macro_f1.tex`, figure `exp2_within_macro_f1`.
- **H2**: `\res{hyp.H2.verdict}` (`\res{hyp.H2.held}` of `\res{hyp.H2.total}` datasets).
  - Best method per dataset: `\res{best.<d>.method}`.
  - XGBoost minus the best text model, e.g. UNSW:
    `\resdiff{exp2.xgboost.none.nf_unsw_nb15.macro_f1}{exp2.distilbert.e3_kv.nf_unsw_nb15.macro_f1}` *(check
    `best.nf_unsw_nb15.distilbert.encoding`)*.
- **H1**: `\res{hyp.H1.verdict}`. E1 vs the best method per dataset, e.g. CIC:
  `\res{exp2.distilbert.e1_ppfle.nf_cse_cic_ids2018.macro_f1}` vs `\res{exp2.xgboost.none.nf_cse_cic_ids2018.macro_f1}`.
- **Error analysis** (`results/error_analysis.md`; appendix figures `perclass_f1_<d>`, `confusion_<d>`):
  - Hard classes, UNSW: `\res{hard.nf_unsw_nb15}`. `generic` reaches
    `\res{perclass.xgboost.none.nf_unsw_nb15.generic}` with XGBoost and
    `\res{perclass.bert_small.e1_ppfle.nf_unsw_nb15.generic}` with small BERT E1.
  - `worms`: XGBoost `\res{perclass.xgboost.none.nf_unsw_nb15.worms}`, DistilBERT E3
    `\res{perclass.distilbert.e3_kv.nf_unsw_nb15.worms}`, small BERT E1
    `\res{perclass.bert_small.e1_ppfle.nf_unsw_nb15.worms}`.
  - CIC `infiltration` predicted as benign: `\res{confusion.xgboost.none.nf_cse_cic_ids2018.infiltration.benign}` %
    (XGBoost).

### 6.3 Exp 3: cross-dataset (RQ3, H3)
- Figure `exp3_heatmaps_grid`, table `generated/exp3_cross.tex`, appendix `exp3_offdiag`.
- Same-dataset vs transfer, e.g. XGBoost: `\res{exp3.xgboost.none.diag.macro_f1}` vs
  `\res{exp3.xgboost.none.off.macro_f1}`.
- Off-diagonal macro-F1 per method, `exp3.<m>.<e>.off.macro_f1`:
  - DistilBERT E2 `\res{exp3.distilbert.e2_ppfle_bins.off.macro_f1}`, XGBoost `\res{exp3.xgboost.none.off.macro_f1}`;
  - small BERT E2 `\res{exp3.bert_small.e2_ppfle_bins.off.macro_f1}` vs E1 `\res{exp3.bert_small.e1_ppfle.off.macro_f1}`.
- **H3**: `\res{hyp.H3.verdict}`. Binning helps both models; readable text helps small BERT only *(check H3 lines in
  `results/analysis_summary.md`)*.
- Recall vs false alarms (`.off.attack_recall`, `.off.fpr`):
  - DistilBERT E2: `\res{exp3.distilbert.e2_ppfle_bins.off.attack_recall}` at FPR
    `\res{exp3.distilbert.e2_ppfle_bins.off.fpr}`;
  - XGBoost: `\res{exp3.xgboost.none.off.attack_recall}` at `\res{exp3.xgboost.none.off.fpr}`.
- Collapsing cells, e.g. XGBoost CIC → BoT `\res{exp3.xgboost.none.nf_cse_cic_ids2018.nf_bot_iot.macro_f1}`.

### 6.4 Pretraining (H4)
- `\res{hyp.H4.verdict}`. DistilBERT minus small BERT:
  - on hashes, e.g. E2 cross-dataset:
    `\resdiff{exp3.distilbert.e2_ppfle_bins.off.macro_f1}{exp3.bert_small.e2_ppfle_bins.off.macro_f1}`;
  - on readable text, e.g. E3 on UNSW:
    `\resdiff{exp2.distilbert.e3_kv.nf_unsw_nb15.macro_f1}{exp2.bert_small.e3_kv.nf_unsw_nb15.macro_f1}`.
- Confound: model size, `\res{params.distilbert.millions}` M vs `\res{params.bert_small.millions}` M parameters.

### 6.5 Exp 4: privacy (RQ4)
- Table `generated/privacy_attack.tex` (+ `privacy_attack_edge.tex`), figure `privacy_attack_overall`, appendix
  `privacy_attack_columns`.
- E1 tokens reversed (min–max over datasets):
  - L1, enumerating `\res{privacy.n_low_cardinality}` columns:
    `\res{privacy.e1_ppfle.enumerate.token_share.min}`–`\res{privacy.e1_ppfle.enumerate.token_share.max}` %;
  - L2, + integers up to `\res{privacy.bruteforce.max}`:
    `\res{privacy.e1_ppfle.bruteforce.token_share.min}`–`\res{privacy.e1_ppfle.bruteforce.token_share.max}` %;
  - L3, + public data:
    `\res{privacy.e1_ppfle.public.token_share.min}`–`\res{privacy.e1_ppfle.public.token_share.max}` %.
- Whole rows reversed at L3:
  - ToN-IoT `\res{privacy.nf_ton_iot.e1_ppfle.public.row_share}` %, CIC
    `\res{privacy.nf_cse_cic_ids2018.e1_ppfle.public.row_share}` %;
  - UNSW `\res{privacy.nf_unsw_nb15.e1_ppfle.public.row_share}` %, BoT `\res{privacy.nf_bot_iot.e1_ppfle.public.row_share}` %.
  - What resists: fractional bytes-per-second values (`\res{privacy.col.e1_ppfle.public.SRC_TO_DST_SECOND_BYTES}` %)
    and large throughput counters (`\res{privacy.col.e1_ppfle.public.DST_TO_SRC_AVG_THROUGHPUT}` %).
- Edge-IIoTset (`privacy_edge.*` keys, levels L1–L2 only, no independent public data): as published
  `\res{privacy_edge.edge_iiotset_raw.e1_ppfle.bruteforce.token_share}` % of tokens,
  `\res{privacy_edge.edge_iiotset_raw.e1_ppfle.bruteforce.row_share}` % of rows; TCP seq/ack numbers resist.
- E2: `\res{privacy.e2_ppfle_bins.enumerate.token_share.min}` % of tokens already at L1 (only
  `\res{cfg.encoders.binning.n_bins}` tokens per binned column). This reveals the bin index, not the value.
- Message: unsalted hashing of low-entropy fields is pseudonymisation, not privacy. A keyed hash (HMAC) blocks the
  dictionary but still leaks when two values are equal.

## 7. Discussion

- Does the encoding matter?
  - Within a dataset, the best encoding changes from dataset to dataset, and everything stays at or below XGBoost
    *(check `hyp.H2.verdict`)*.
  - Across datasets, binning beats raw hashing for both language models *(check `hyp.H3.verdict`)*.
- Cost: median training run XGBoost `\res{time.xgboost.train.median_s}` s, small BERT
  `\res{time.bert_small.train.median_min}` min, DistilBERT `\res{time.distilbert.train.median_h}` h (laptop GPU).
- Hashing removes numeric order. Binning restores coarse order and shares tokens across datasets, which explains the
  transfer result.
- PPFLE gives neither privacy (Exp 4) nor an accuracy edge (Exp 2/3).

## 8. Limitations

- DistilBERT at `\res{seeds.distilbert}` seed. Small BERT seed spread is large on CIC, e.g.
  `\res{exp2.bert_small.e1_ppfle.nf_cse_cic_ids2018.macro_f1.std}`.
- Pilot tuning on one dataset.
- Small BERT is smaller than the paper's model (`\res{params.bert_small.millions}` M).
- E1/E2 cut for DistilBERT (side runs in §5).
- UNSW host fingerprint kept on purpose. Edge results are on the fixed data.
- One sample per dataset. Timings mix laptop power states.
- Exp 4 on Edge-IIoTset has no public-dictionary level (lower bound).

## 9. Conclusion

- Three to four sentences: the replication verdict, the XGBoost baseline, binning for transfer, PPFLE not private.

## Checklist before submitting

- [x] Edge run finished → `uv run ftw analysis`; the log lists any key in this outline or the report without a value.
- [ ] Re-read `results/analysis_summary.md`; verdicts can change with new results, and every *(check …)* statement
      above depends on one.
- [ ] If more DistilBERT seeds are run, rerun the analysis. The † marks and "tentative" tags disappear by themselves.
- [ ] Search the text for digits typed by hand (`grep -nE '[0-9]' report/sections/*.tex`); there should be none in the
      text.
