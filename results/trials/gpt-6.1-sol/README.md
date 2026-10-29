# Harbor trials: Claude Code / GPT-6.1 Sol (xhigh; three, all failed)

**Execution framework:** Harbor. **Agent harness:** Claude Code (`claude-code`). **Model:** `gpt-6.1-sol`. **Reasoning effort:** `xhigh`. Each trial used the packaged task instruction and agent environment, with a 2-hour agent timeout, and did not use the web search tool. Each delivered `predict.py` was graded in the packaged verifier image.

I ran these trials on a temporary VM and imported the results into this repository. The original VM trial directories and agent transcripts are no longer available.

For new trials through Harbor, use [configs/run_trials.sh](../../../configs/run_trials.sh).

Each trial ended by itself with a finished `predict.py` and a closing report; none reached the 2-hour limit. The runs were made one after another on 2026-10-28.

| Trial | Start (UTC) | Duration | Winter (share of day-features in tolerance; pass mark 0.92 plus weekly rules) | Spring | Reward |
|---|---|---|---|---|---|
| 1 | 2026-10-28 20:52 | 93 min | pass (0.940) | **fail (0.728)**, 9 days with more than 2 of 8 features out | 0 |
| 2 | 2026-10-28 22:32 | 103 min | pass (0.949) | **fail (0.652)**, 16 days | 0 |
| 3 | 2026-10-29 00:23 | 102 min | pass (0.968) | **fail (0.705)**, 12 days | 0 |

**Where they failed.** All three passed winter and failed spring. In spring the indoor temperatures missed most often: evening temperature was in tolerance on only 32-36 % of days, night 32-46 %, daytime 39-71 % and the daily minimum 43-71 %. Daily energy missed on 7-18 % of days. The misses were worst in the first two weeks, with weekly shares down to 0.571, 0.464 and 0.518. Weekly energy and weekly peaks stayed within their limits.

**What they said.** All three closing reports gave their confidence as "moderate" and named severe winter cold as the main uncertainty; none named spring.

Per trial: `submission/predict.py` (the delivered file, unmodified; it reads no other file), `verifier/test-stdout.txt` (that file alone graded in the packaged verifier image), `agent_final_report.md` (the agent's closing message, verbatim) and `run.json` (model, settings, times, event counts and token usage, and the packaged verifier's results).
