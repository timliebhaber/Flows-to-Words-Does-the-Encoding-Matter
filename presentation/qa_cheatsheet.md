# Q&A cheat sheet (station session)

Short answers to the questions visitors are most likely to ask. Numbers come from `results/*.md` (generated). The
report has the full story.

**1. Why turn a table into text at all?**
Language models are becoming the default tool for security text (logs, alerts, reports). If network flows are text
too, one model can read everything, and pretrained knowledge could help. SecurityBERT showed this works on one
dataset. We test *how* the table is written and whether that changes the result.

**2. What exactly are the four encodings?**
E1 (the paper's PPFLE): one SHA-256 hash of `column$value` per feature, cut to 16 hex characters. E2: the same, but
numbers are first put into one of 10 quantile bins fitted on the training split. E3: `dst port=80 proto=6 …`. E4:
an English sentence template using all 47 features. The demo's tab 2 shows all four for one flow.

**3. Why does XGBoost win?**
The features are numbers, and trees split on numeric thresholds directly. A text model first has to learn what
"80" or a hash means from the tokens. Within one dataset XGBoost is as good as or better than the best text model on
all 5 datasets (H2). On ToN-IoT, BoT-IoT, CIC and Edge the gap is within our tie margin. On UNSW-NB15, XGBoost leads
by about 0.07 macro-F1.

**4. So are language models useless here?**
No. Across networks (train on one dataset, test on another) DistilBERT with binned hashes (E2) is the best method in
our grid: 0.62 mean off-diagonal macro-F1, against 0.55 for XGBoost and 0.49 for the paper's E1. That is one seed,
so it is a tentative result.

**5. Why do binned hashes transfer better?**
Bins hide the exact values that differ from network to network (exact byte counts, durations). The model learns
"large / small" instead of memorising numbers that only exist in one network.

**6. What is the Edge-IIoTset leak?**
In the published CSV, empty text fields are written "0" in files from normal traffic and "0.0" in files from attack
traffic. A model can read the label from the spelling. XGBoost gets 1.000 binary accuracy on the published data and
0.973 once the spelling is unified. Small BERT E1 goes from 0.978 to 0.955. Galal (2026) found the same artifact
independently. Tab 5 of the demo shows the spelling shares.

**7. Did you replicate the 98.2 %?**
Yes, on the data as published: 0.978 ± 0.003 accuracy vs 0.982 in the paper (within our 0.03 margin). But part of
that number comes from the leak above.

**8. How does the hash attack work? Isn't SHA-256 secure?**
SHA-256 is fine, but the input is tiny and predictable. A port is one of 65,536 values, a protocol one of 256, and a
byte count is an integer. An attacker who knows the scheme (it is in the paper) hashes every guess and looks it up.
No secret is involved. Recovered E1 values: 95–99.8 % of all hashed tokens per dataset, every one checked against the
true value (0 wrong). About 30 s per dataset on a laptop CPU.

**9. What would make it private?**
A secret key (HMAC or salted hashing) stops this dictionary attack. But equal values still give equal tokens, so
frequency analysis still works. Real privacy needs things like differential privacy or keeping the data on-site.
Hashing alone is pseudonymisation, not privacy.

**10. Why is the cross-dataset grid binary only?**
The four NetFlow datasets name their attacks differently (e.g. "scanning" vs "reconnaissance"), so multi-class
labels don't line up. "Attack or benign" is the same question everywhere.

**11. How did you avoid cheating / leakage?**
We drop IPs, source port and timestamps before anything else, dedupe, then do a 70/15/15 stratified split. Bins,
tokenizers and scalers are fitted on train only, tuning uses validation only, and the test set is used once at the
end. A single-feature XGBoost check flags any feature that alone gives > 99 % macro-F1.

**12. Why only one seed for DistilBERT?**
Time: one seed of all DistilBERT runs took ~65 h on our laptop GPU. XGBoost and small BERT have 3 seeds. Verdicts that
use DistilBERT are marked tentative in the report. More seeds are being run on the university cluster.

**13. What's macro-F1 and why not accuracy?**
The F1 score per class, averaged with equal weight. Most flows are benign, so a model that ignores rare attacks still
gets high accuracy. Macro-F1 punishes that. We report accuracy too, to compare with the paper.

**14. How big are the models / data?**
Small BERT: 4 layers, hidden size 128, about 1.5 M parameters, BPE vocabulary 5,000 trained per encoding. DistilBERT:
66 M parameters, pretrained. Data: 200k sampled flows per dataset, ~99k–134k after deduplication.

**15. What would you do next?**
More DistilBERT seeds. Ablations: drop the destination port, and use 5 / 10 / 20 bins. Compare with prompting a
large LLM directly. And test a keyed (HMAC) encoding to see whether privacy costs accuracy.
