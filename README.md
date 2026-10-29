# House Thermal Model

A heat-pump home can behave quite differently in January than it does in October. Heating capacity falls as outdoor temperatures drop, the thermostat's morning recovery can trigger backup heat, and spring sunlight changes the balance between heat loss and indoor comfort. A forecast that fits one season can still miss the next.

I built this Terminal-Bench 3 task to test that problem directly: use 75 days of autumn electricity and thermostat readings to predict the same house through an unseen winter and spring. The deliverable is a working Python predictor, evaluated on both electrical demand and indoor temperature.

I am [Sam Zaheer](https://github.com/SamZaheer). The runnable package is in [`tasks/house-thermal-model/`](tasks/house-thermal-model/); its [reviewer notes](tasks/house-thermal-model/README.md) explain the reference model and verification design. I packaged it against `harbor-framework/terminal-bench` commit `1dcda871` and Harbor `0.23.1.dev202609170426`.

## Start with the autumn data

The workspace contains measurements for an all-electric house in Oklahoma City, covering 28 September through 11 December 2020 at five-minute intervals:

- [`weather_autumn.csv`](tasks/house-thermal-model/environment/data/weather_autumn.csv): temperature, humidity, sunlight, wind and pressure.
- [`meter_thermostat_autumn.csv`](tasks/house-thermal-model/environment/data/meter_thermostat_autumn.csv): whole-house electrical demand and the thermostat's temperature reading.
- [`house.json`](tasks/house-thermal-model/environment/data/house.json): location, time zone and a short description of the building and heating system.
- [`predict.py`](tasks/house-thermal-model/environment/predict.py): a flat-forecast stub to replace.

The submitted program must accept a new weather file and write one prediction for every timestamp:

```bash
python3 /app/predict.py WEATHER.csv OUT.csv
# OUT.csv columns: time,power_kW,indoor_temp_C
```

The agent has two hours to develop its solution and may access the internet. Prediction runs offline with Python 3.12, numpy, scipy and pandas, with a ten-minute limit per weather file. Supporting files under `/app` are retained. The full contract is in [instruction.md](tasks/house-thermal-model/instruction.md).

The weather comes from the Open-Meteo historical archive, based on ERA5 and licensed under CC BY 4.0. I simulated the building, its equipment, occupants and meter readings on that weather. This gives the verifier a known answer without exposing a real household's records. [Data provenance](DATA-SOURCES.md) and [weather attribution](NOTICE.md) describe the source files and transformations.

## What changes after autumn

**Winter tests the heating system.** The observed autumn bottoms out at −3.1 °C; the graded winter reaches −10.7 °C. At those lower temperatures, heat-pump capacity, defrost cycles and resistance backup affect both demand and the house's ability to recover from an overnight setback.

**Spring tests the heat balance.** Higher sun angles change gains through the windows, while a deciduous tree shades the south-facing windows as it comes into leaf. A model fitted to autumn must account for that seasonal change as well as the building's stored heat and the occupants' schedule.

The task instruction describes the forecast goal without supplying these mechanisms. Inferring how they interact is the main challenge. The only available measurements are from autumn, so a good fit to those measurements cannot serve as an independent check of winter or spring performance. This resembles the work of a building-energy engineer or utility analyst forecasting seasonal consumption and peak load from smart-meter telemetry.

My [reference implementation](tasks/house-thermal-model/solution/predict.py) uses the same house model that generated the data. It tracks heat in the room air and furnishings, internal mass and envelope, together with solar gains, infiltration, thermostat control and heat-pump operation. Household loads and sensor noise differ between the reference and the hidden truth, so even the reference is not an exact copy of the target series.

## How a forecast earns a pass

The grader checks the consequences of the predicted time series for energy use, peak load and comfort. Each day contributes eight measurements:

| Measurement | Allowed error |
|---|---|
| Daily electricity use | 10%, with a 2.5 kWh floor |
| Highest 15-minute demand | 20%, with a 1.5 kW floor |
| Hours above 7.5 kW | 30%, with a 0.75-hour floor |
| Morning recovery temperature, daily minimum, night and evening temperatures | 0.6 °C each |
| Daytime temperature | 0.9 °C |

The morning window is 05:30–09:30; night is 00:00–05:00, evening 18:00–22:00 and daytime 10:00–16:00. For relative tolerances, the grader uses the larger of the percentage allowance and the absolute floor.

Both seasons must satisfy all of these conditions: at least 92% of daily measurements within tolerance overall, at least 80% in each graded week, and no day with more than two misses. Weekly energy must be within 12% (15 kWh floor), and the weekly mean of daily peaks within 30% (2 kW floor). The [grading code](tasks/house-thermal-model/tests/housegrade.py) defines the calculations and week boundaries.

The first three days of each weather file are unscored warm-up. Winter scoring covers 25 January–11 February and 20–28 February 2021. I excluded 12–19 February, when the daily mean reached roughly −19 °C, to avoid making the test depend on an extreme cold snap far beyond the autumn observations. Every timestamp still needs an output row; values on unscored days are unrestricted. Timestamps with an explicit offset are converted to the data's UTC−6 standard time.

The hidden weather and answers live in a separate verifier image under root-only `/tests`. Harbor transfers the submitted `/app` tree, excluding the three supplied autumn data files; the verifier restores pristine copies of those files. It then runs the predictor as `nobody`, offline, in a fresh working directory for each season. A 600-second watchdog and a 4 GB container limit bound execution. After prediction, remaining processes running as `nobody` are killed, and root accepts output only from a regular file owned by that user. The reward directory remains root-only. [Verifier checks](results/verifier-checks.md) record the isolation and exploit tests.

## Calibration before interpretation

I checked whether the tolerances admit plausible solutions recovered from the autumn measurements, and whether omitting important physics still produces a failure. A share below is the fraction of daily measurements within tolerance; the weekly and per-day conditions also apply.

| Calibration experiment | Winter | Spring |
|---|---|---|
| Reference model | Pass, 0.995 | Pass, 0.982 |
| Ten parameter draws, each perturbed by up to ±2% | All pass, 0.977–1.000 | All pass, 0.969–0.982 |
| Seventeen fits of the model structure to autumn data, from textbook starting values | All pass, 0.940–0.981 | All pass, 0.960–0.987 |
| Twelve fits with a mechanism omitted or a plain RC model substituted | Nine fail | Nine fail; none of the twelve passes both seasons |
| Gradient boosting fitted to autumn | Not computed on this window | Fail, 0.763 |
| Unmodified stub | Fail, 0.255 | Fail, 0.290 |

The omission experiments cover thermostat setbacks, a second thermal mass, tree shading and backup heating. Removing backup heat, setbacks or the second mass breaks winter performance; removing the tree, setbacks or the second mass breaks spring performance. The plain RC model fails both seasons. The gradient-boosting baseline fits autumn well (0.97 of daily measurements within tolerance) but does not carry that performance into spring.

The [fairness audit](tools/fairness-audit/grading-variants.json) records every candidate under the grading variants considered, including weekly errors and daily misses. All six model submissions below fail every variant. [Shortcut baselines](tools/fairness-audit/shortcut-baselines.txt) provide additional comparisons on the full winter window.

## Recorded Harbor trials

I ran both sets of trials on a temporary VM through Harbor, using the Claude Code (`claude-code`) harness with the packaged task: Claude Opus 5.5 at `max` reasoning effort and GPT-6.1 Sol at `xhigh`. Every trial had a two-hour agent limit, and I graded the delivered predictors in the packaged verifier.

| Model and effort | Winter shares, trials 1 / 2 / 3 | Spring shares, trials 1 / 2 / 3 | Time to finish | Outcome |
|---|---|---|---|---|
| Claude Opus 5.5, `max` | 0.912 / 0.889 / 0.898 | 0.652 / 0.741 / 0.915 | 15–18 min | All three fail both seasons |
| GPT-6.1 Sol, `xhigh` | 0.940 / 0.949 / 0.968 | 0.728 / 0.652 / 0.705 | 93–103 min | All three pass winter and fail spring |

The Opus predictors included substantial physical models, but their winter peaks and energy estimates still missed the limits. Spring failures across both models concentrated on indoor temperatures, especially in the first two weeks, with additional daily energy misses. All six runs ended on their own before the time limit. Their closing reports emphasized winter cold as a risk; the Sol reports did not identify spring as the unresolved issue. Detailed outcomes, delivered predictors and closing messages are in the [Opus records](results/trials/opus-5.5/) and [Sol records](results/trials/gpt-6.1-sol/).

For the adversarial check, the Opus Harbor trial received the unchanged [TB3 CI hack prompt](configs/hack-trial-prompt.md) after the task instruction. It reported no bypass and left the stub; the verifier awarded 0. The GPT-6.1 Sol (`xhigh`) Harbor adversarial attempt, also using the Claude Code harness on the temporary VM, was refused by the provider before any agent action and was not graded. The [adversarial records](results/cheat-trials/) retain those outcomes.

Local package validation used macOS arm64 and Docker 29.8 on Colima (8 CPU, 12 GB). The saved evidence includes [26 passing static checks](results/checks/static/), successful image builds, [oracle reward 1](results/execution/oracle/verifier/test-stdout.txt), [nop reward 0](results/execution/nop/verifier/test-stdout.txt), five rejected exploit predictors and five accepted output variants. The [implementation review](results/checks/implementation/check-result.json) recorded 34 passes, no failures and one item marked not applicable; [run details](results/checks/implementation/run.json) accompany that report.

## Run or inspect the package

Use the repository root for these commands. Static checks require a terminal-bench checkout and Python 3.11 or newer:

```bash
for check in /path/to/terminal-bench/scripts/checks/check-*.sh; do
  bash "$check" tasks/house-thermal-model
done
```

The task can be exercised without calling a model:

```bash
harbor run -p tasks/house-thermal-model --agent oracle --env docker
harbor run -p tasks/house-thermal-model --agent nop --env docker

# Re-score one saved submission.
docker build -t hphouse-verifier tasks/house-thermal-model/tests
docker run --rm --network none --cpus 2 --memory 4g \
  -v "$PWD/results/trials/opus-5.5/trial-1/submission":/sub:ro hphouse-verifier \
  bash -c 'cp -r /sub/. /app/ && /tests/test.sh; cat /logs/verifier/reward.txt'
```

To start new model trials, configure the credentials and model routing for your Harbor environment, then use the [launcher](configs/run_trials.sh):

```bash
bash configs/run_trials.sh opus   # three Claude Code / Opus 5.5 max attempts
bash configs/run_trials.sh sol    # three Claude Code / GPT-6.1 Sol xhigh attempts
bash configs/run_trials.sh opus --n-attempts 1 \
  --extra-instruction-path configs/hack-trial-prompt.md
```

For a code review, start with [`task.toml`](tasks/house-thermal-model/task.toml), the [house simulator](tasks/house-thermal-model/solution/housemodel/), and the [verifier entry point](tasks/house-thermal-model/tests/test.sh). [`tools/truth/make_truth.py`](tools/truth/make_truth.py) regenerates the simulated series from the model, raw weather and fixed seeds. [`tools/weather/`](tools/weather/) holds the original Open-Meteo downloads.

## My experience with the underlying work

I am a Senior AI/ML Engineer with more than eight years of experience in machine learning, data engineering and production AI systems. I earned a Bachelor of Science in Electrical Engineering from Oklahoma State University.

At The Data Sherpas, I have directed AI architecture for time-series forecasting, predictive maintenance, anomaly detection and optimization across more than 20 renewable-energy sites. My production ML work includes feature engineering, model evaluation and inference pipelines supporting more than 100 million telemetry events per month. That background informs the questions I focused on here: what telemetry can identify about a physical system, how seasonal changes expose model errors, and whether the evaluation accepts more than one successful calibration.
