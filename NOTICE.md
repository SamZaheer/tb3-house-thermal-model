# Notice on third-party data in this repository

**Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).** The Open-Meteo historical weather archive is based on ERA5 reanalysis data from the Copernicus Climate Change Service (C3S).

Where it is used:
- `tools/weather/okc_autumn.json`, `okc_winter.json`, `okc_spring.json`: hourly weather for Oklahoma City as downloaded from the Open-Meteo historical weather API, unmodified.
- `tasks/house-thermal-model/environment/data/weather_autumn.csv` (and its copy in `tests/given/`), `tests/hidden/weather_winter.csv` and `tests/hidden/weather_spring.csv`: derived from those files. **Changes made:** times shifted from UTC to local standard time (UTC-6); temperature, dew point, wind speed and pressure interpolated from hourly to 5-minute values; global and diffuse radiation turned into 5-minute global, direct-normal and diffuse irradiance through a clear-sky index; relative humidity recomputed from temperature and dew point; values rounded. The code is `load_openmeteo`, `to_5min` and `public_columns` in `tasks/house-thermal-model/solution/housemodel/weather.py`.

Everything else in this repository is my own work: the house, its heat pump and occupants, the smart-meter and thermostat data (simulated by me on the weather above), the grader, the reference solution, the task text and the results. No real household's data is used.
