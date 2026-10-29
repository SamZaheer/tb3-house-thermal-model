"""Regenerate the task's data from my house model and the raw weather.

    python3 tools/truth/make_truth.py OUT_DIR

Writes, for the autumn the agent is given and for the two graded periods, the
weather file (public columns) and the 5-minute meter and thermostat series:
  weather_autumn.csv, meter_thermostat_autumn.csv   (environment/data/)
  weather_{winter,spring}.csv, truth_{winter,spring}.csv   (tests/hidden/)
The model is the one in solution/housemodel/; each period is one run of it with
its own seed for the household loads and the thermostat noise. house.json is
written by hand and not regenerated here.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tasks", "house-thermal-model", "solution", "housemodel"))
import house  # noqa: E402
import weather  # noqa: E402

WDIR = os.path.join(REPO, "tools", "weather")
# name: (raw weather file, first day, day after the last, seed, output names)
PERIODS = {
    "given": ("okc_autumn.json", "2020-09-28", "2020-12-12", 11, ("weather_autumn.csv", "meter_thermostat_autumn.csv")),
    "winter": ("okc_winter.json", "2021-01-22", "2021-03-01", 12, ("weather_winter.csv", "truth_winter.csv")),
    "spring": ("okc_spring.json", "2021-03-26", "2021-04-26", 13, ("weather_spring.csv", "truth_spring.csv")),
}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, (raw, start, end, seed, (wname, tname)) in PERIODS.items():
        w = weather.to_5min(weather.load_openmeteo(os.path.join(WDIR, raw)), start, end)
        series, _, _ = house.simulate(w, seed=seed)
        weather.public_columns(w).to_csv(os.path.join(out_dir, wname), date_format="%Y-%m-%d %H:%M")
        series.to_csv(os.path.join(out_dir, tname), date_format="%Y-%m-%d %H:%M")
        print(name, len(series), "rows")


if __name__ == "__main__":
    main(sys.argv[1])
