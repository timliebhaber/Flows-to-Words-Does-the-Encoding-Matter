# Nov 4 station session: pitch + demo

The instructor's format: a short pitch (**Problem → Solution → Benefit → Action**), then a working demo on a laptop.

| File | What it is |
|---|---|
| [`pitch_script.md`](pitch_script.md) | ~2.5 min spoken script, split into 4 parts for 3 speakers, with the source of every number |
| Slides (3) | claude.ai deck: https://claude.ai/artifact/DAF1YGcwg1tf8P2eS9kpdP (download as PPTX/PDF from the page; private until Daniel shares it) |
| [`demo_runbook.md`](demo_runbook.md) | How to start the demo app, the 5-step visit (~3 min), what to do if something breaks |
| [`qa_cheatsheet.md`](qa_cheatsheet.md) | 15 likely visitor questions with short answers |

Demo app: `uv run ftw demo` once (~6 min GPU), then `uv run streamlit run src/demo/app.py -- --config configs/demo.yaml`.

**If results change** (e.g. the DistilBERT seeds 1–2 from the cluster), rerun `uv run ftw analysis`. The app's findings step
updates by itself. The numbers typed into the script, the slides and the Q&A do not, so check them against
`results/analysis_summary.md` and `results/privacy_attack.md`.
