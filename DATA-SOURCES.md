# Data sources and licence status

**Short version: the weather is real and comes from the Open-Meteo historical weather archive (ERA5-based), licensed CC BY 4.0, with attribution in [NOTICE.md](NOTICE.md). I simulated the house, its equipment and its meter and thermostat data.**

## Weather

| | |
|---|---|
| Provider | Open-Meteo historical weather API (`https://archive-api.open-meteo.com/v1/archive`), reanalysis based on ERA5 (Copernicus Climate Change Service) |
| Licence | CC BY 4.0; attribution "Weather data by Open-Meteo.com" |
| Location | Oklahoma City, OK, USA; the archive's grid point is 35.466 N, 97.50 W, elevation 367 m |
| Variables (hourly) | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `wind_speed_10m` (m/s), `surface_pressure` |
| Downloaded | 2026-10-21 (file dates) |

| Raw file (`tools/weather/`) | UTC range | Used for | Task file (local standard time, 5-minute) |
|---|---|---|---|
| `okc_autumn.json` | 2020-09-20 to 2020-12-15 | the data the agent sees | `environment/data/weather_autumn.csv`: 2020-09-28 to 2020-12-11 |
| `okc_winter.json` | 2021-01-15 to 2021-03-05 | hidden winter period | `tests/hidden/weather_winter.csv`: 2021-01-22 to 2021-02-28 |
| `okc_spring.json` | 2021-03-20 to 2021-04-30 | hidden spring period | `tests/hidden/weather_spring.csv`: 2021-03-26 to 2021-04-25 |

The conversion from the raw files to the task files is described in [NOTICE.md](NOTICE.md).

## Meter and thermostat data

I simulated all of it with the house model in `tasks/house-thermal-model/solution/housemodel/` on the weather above: a fictional 165 m2 single-storey house of 1985 with a variable-speed heat pump and backup resistance heat, three occupants on a weekday and weekend schedule, randomised household loads and thermostat sensor noise. The autumn series the agent sees and the hidden winter and spring series use different random draws of the household loads. No real household, meter or utility data is used.
