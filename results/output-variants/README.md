Honest submissions that the verifier must accept, run in the packaged verifier image on 2026-10-26. Each folder holds a `predict.py` that wraps the reference: copy `tasks/house-thermal-model/solution/predict.py` beside it as `ref_predict.py` and `solution/housemodel/` as `housemodel/`, then grade the folder as in [../verifier-checks.md](../verifier-checks.md).

| Folder | What the submission does | Result |
|---|---|---|
| `data_params/` | reads a file it keeps in `/app/data/` (`fitted_params.json`) | reward 1 |
| `nan_spinup/` | writes NaN for the three spin-up days | reward 1 |
| `tz_local/` | writes times with a UTC offset, `2021-01-22T00:00:00-06:00` | reward 1 |
| `tz_utc/` | writes the same instants in UTC, `2021-01-22T06:00:00+00:00` | reward 1 |
| `link_own/` | writes its forecast to `/tmp` and makes `OUT.csv` a symbolic link to it | reward 1 |

Every one scores what the reference scores: winter 0.995, spring 0.982 of day-features in tolerance.
