# Harbor trials: Claude Code / Claude Opus 5.5 (max; three, all failed)

**Execution framework:** Harbor. **Agent harness:** Claude Code (`claude-code`). **Model:** `claude-opus-5-5`. **Reasoning effort:** `max`. Each trial used the packaged task instruction and agent environment, with a 2-hour agent timeout.

I ran these Harbor trials on a temporary VM and imported the results into this repository. The original VM trial directories and agent transcripts are no longer available.

For new trials, use [configs/run_trials.sh](../../../configs/run_trials.sh).

Each run ended by itself with a finished `predict.py` and a closing report; none reached the 2-hour limit. Trials 2 and 3 ran at the same time.

| Trial | Start (UTC) | Duration | Winter (share of day-features in tolerance; pass mark 0.92 plus weekly rules) | Spring | Reward |
|---|---|---|---|---|---|
| 1 | 2026-10-21 18:15 | 16 min | **fail (0.912)** | **fail (0.652)**, 14 days with more than 2 of 8 features out | 0 |
| 2 | 2026-10-21 18:31 | 15 min | **fail (0.889)**, 4 days with more than 2 out | **fail (0.741)**, 10 days | 0 |
| 3 | 2026-10-21 18:32 | 18 min | **fail (0.898)**, 2 days with more than 2 out | **fail (0.915)** | 0 |

Scores are from the packaged verifier, with the winter window of the task (12-19 February left out). Local re-scoring of the imported submissions gave the same numbers.

**Where they failed.** In spring it is mostly the indoor temperature (evening, night, daytime and minimum) and, on about a quarter of the days, daily energy; worst in the first two weeks (late March to early April), with weekly shares down to 0.46 (trial 1) and 0.64 (trial 2). In winter, trial 1 misses the 15-minute peak on 41 % of days; trials 2 and 3 miss daily energy (78 % and 70 % of days in tolerance) and the 15-minute peak (56 % and 74 %), with weekly energy 10 % low in the last week.

**What they said.** All three closing reports were cautious: trial 1 "moderately confident it will pass", trial 2 "moderately confident for spring and less so for winter", trial 3 "not confident it meets the 'nearly all measures, in every week' bar". Each named cold below the autumn's -3 C as the main risk; spring appeared only as a risk for cooling, which the autumn shows on few days.

Per trial:

- `submission/`: the files the agent delivered, unmodified (`predict.py`; trial 2 also `model_params.json`, which its `predict.py` reads).
- `verifier/test-stdout.txt`: the submission graded in the packaged verifier image.
- `agent_final_report.md`: the agent's closing message, verbatim.
- `run.json`: model, settings, timestamps, duration, retained tool-call counts, and the packaged verifier's results.

The records here are imported result summaries, submissions and verifier outputs.
