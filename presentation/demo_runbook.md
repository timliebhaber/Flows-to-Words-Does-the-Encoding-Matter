# Demo runbook (Nov 4 station session)

Five steps, each with something to do, in standard terminology (flow, feature, hash, token, classifier)
with a short explanation where needed. All visitor-facing text lives in the `story` section of
`configs/demo.yaml`.

## The day before
1. `uv sync` (installs streamlit, which is in the dev group).
2. `uv run ftw demo --dry-run` should say `demo_models 1/1 done`. If not: `uv run ftw demo` (~6 min on the laptop GPU).
   The models go to `data/demo_models/` (gitignored), so a teammate's laptop needs its own run, plus the processed
   data and tokenizers.
3. Start the app once (below) and click through all 5 steps, including **Run the attack on this flow** (~16 s), so the
   caches are warm before the first visitor.
4. Take screenshots of every step as a backup.

## At the station
```bash
systemctl --user stop jarvis          # frees 3.2 GB of GPU memory (the app also runs on CPU)
cd ~/Github/NLProfessionals
uv run streamlit run src/demo/app.py -- --config configs/demo.yaml
```
The app opens at http://localhost:8501. Use full screen (F11); `Ctrl +` / `Ctrl -` to fit the projector.
`http://localhost:8501/?step=3` opens a given step directly. Stop the app with `Ctrl+C` in the terminal.

## Flow of a visit (~3 min)
1. **The flow.** "A network sensor writes one record per connection: 47 features." Press *Draw a random flow*:
   client → destination service, packets and bytes each way, duration. The label stays hidden until step 3.
2. **Flow → text.** "A language model reads text, so the row has to be serialised." The original flow is shown on
   top and encoded live, feature by feature, into all four texts (E1 PPFLE hashes, E2 binned PPFLE, E3 key=value,
   E4 sentence template); the panel in the middle shows how the current feature becomes each token
   (`sha256("L4_DST_PORT$6437")[:16]`, bin label, `dst port=6437`, …). *Encode again* replays it. Afterwards the
   feature chips highlight one feature in the grid and in all four texts. Optional: *Show BPE tokens*.
3. **Classification.** The visitor guesses the class (benign or which attack). Then XGBoost and small BERT on E1
   and E3 predict it, with their top-3 probabilities; the scoreboard keeps count. *Draw another flow* for another
   round. Curious visitors: open **What-if analysis**, raise `IN_PKTS` or change `L4_DST_PORT`, and watch the
   predictions change.
4. **Hash attack.** Show the hash recipe, let the visitor guess the hashed port (common ports are one click), then
   *Brute-force every port* (all 65,536 in a fraction of a second). Finish with *Run the attack on this flow*: the
   47 tiles turn green one by one, each with its attack level (L1–L3), every value verified against the truth.
5. **Results.** Four cards, one per experiment (numbers come from the report, so they are always current), and the
   quiz: "two rows from the published Edge-IIoTset file, which one is the attack?" The spelling "0" vs "0.0" gives
   it away.

## If something goes wrong
- *"Demo models not trained yet"* in step 3 → `uv run ftw demo`, then restart the app.
- App hangs or shows an error → `Ctrl+C`, start again, reload the page.
- GPU busy or out of memory → nothing to do: with `device: auto` the app falls back to CPU (`configs/demo.yaml`).
- A findings card shows "…" → the report numbers are missing: `uv run ftw analysis privacy`.
- No laptop at all → show the screenshots plus the report PDF.
