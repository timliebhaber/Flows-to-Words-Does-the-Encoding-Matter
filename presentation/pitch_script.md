·# Pitch script (Nov 4 station session)

Format from the instructor: **Problem → Solution → Benefit → Action**, 2–3 minutes. About 380 words, so roughly 2.5 minutes at a calm pace.
The split between speakers is only a suggestion. Show the 3-slide deck (link in `presentation/README.md`) while you talk, then turn to the demo laptop.

Numbers are copied from the generated results (source next to each line), so check them again if results change:
`results/analysis_summary.md`, `results/exp1_replication.md`, `results/privacy_attack.md`, `results/dataset_stats.md`.

---

## 1 · Problem (Speaker A, ~35 s) · slide 1

> Every network writes millions of flow records: a big table of ports, bytes, packets and durations.
> Security tools are moving to language models, and one popular idea is to turn each row into *text*, so that a BERT
> can read it.
>
> The best-known example is SecurityBERT. It hashes every value into a token, calls that "privacy-preserving", and
> reports **98.2 % accuracy**.
>
> But two questions were never asked. **Does the way you turn a table into words actually matter?** And **are those
> hashes really private?**

## 2 · Solution (Speaker B, ~45 s) · slide 2

> We built a benchmark to answer both.
> We take the same network flow and write it **four ways**: hashes like the paper, hashes of *binned* values, plain
> `key=value` text, and an English sentence.
>
> Each version trains a small BERT from scratch and a pretrained DistilBERT, and we compare everything against
> plain **XGBoost** on the raw table.
>
> Five public datasets, leak-free splits, **413 runs**, three seeds. *(results.csv rows)* We test within one network, across
> networks (train on one, test on another), and finally we **attack the hashes**.

## 3 · Benefit (Speaker C, ~55 s) · slide 3

> Three findings.
>
> **One: the 98.2 % replicates, but partly by accident.** We get 97.8 %. *(exp1, as published)* In that dataset, empty
> fields are written "0" in normal traffic and "0.0" in attack traffic, so the label leaks through the spelling.
> XGBoost reaches 100 % on that alone. Fix the spelling and BERT drops to 95.5 %. *(exp1, fixed)*
>
> **Two: within one network, plain XGBoost is as good as or better than every language model**, on all five
> datasets. *(H2)* Text pays off when you move to a *new* network: binned hashes transfer best. DistilBERT reaches 0.62
> macro-F1 there, against 0.49 for the paper's encoding and 0.55 for XGBoost. *(H3 + context)*
>
> **Three: the hashes are not private.** A dictionary attack on a laptop recovers 95 to almost 100 % of the hashed
> values. On two of the datasets that is **88 to 93 % of all flows, completely**. *(privacy_attack.md, brute-force
> level: ToN-IoT 87.6 %, CIC 93.1 %)*
>
> So if you build language-model intrusion detection, you now know which encoding to pick, when a tree model is
> enough, and that hashing is not privacy.

## 4 · Action (Speaker A or B, ~20 s) · slide 3, then point at the laptop

> Come try it at our station. **Pick a real network flow, watch it turn into four texts, see which model catches the
> attack, and watch us crack its "private" hashes live.**
>
> And if you work with tables and language models: check your data for leaks, bin your numbers, and don't call
> hashing privacy.

---

### Delivery tips
- Slow down on the three bold findings. They are what people remember.
- Point at the big number on each card of slide 3 as you say it (100% · 5 of 5 · 93%).
- If time is short, cut the DistilBERT numbers in finding two ("binned hashes transfer best" is enough).
