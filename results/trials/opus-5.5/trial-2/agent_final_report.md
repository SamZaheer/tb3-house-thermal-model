I've written `predict.py`, a calibrated simulation of the house. On the autumn data, the weekly figures I checked mostly match. But all of that is in-sample (no held-out validation), and extreme cold is beyond what the data covers. So I'm moderately confident for spring and less so for winter.

**What it does.** It reads the weather file and simulates the house step by step, then writes `time,power_kW,indoor_temp_C` for every row. A 75-day file runs in about 0.6 s. The fitted parameters are built into the script; `model_params.json` next to it holds the same values and overrides them if present, so either works alone.

The model has five parts, each based on what the data showed:
- **Building:** a simple two-temperature model (indoor air plus the slab and structure), with heat from the sun and from appliances.
- **Thermostat:** 21 °C when occupied and 17 °C at night, plus 17 °C on weekdays from 08:00 to about 15:00. Before each occupied period the setpoint ramps up over about 90 minutes. Cooling holds around 26 °C.
- **Heat pump:** variable speed, with output and efficiency falling as it gets colder outside and when it runs flat out. A 10 kW resistance heater kicks in if the house falls well behind.
- **Defrost spikes:** the 12–15 kW spikes in the data are defrost cycles (about 10 kW for about 7.5 minutes). They only happen when it's cold and humid; on the dry −3 °C day there were none. The model predicts them from outdoor temperature and humidity: 56 simulated events vs 59 real ones, and only one day where it got wrong whether any happened.
- **Everything else:** a constant 0.15 kW background load, the fridge cycle (its timing matched the data exactly), a small heater that switches on below 10 °C outdoors, and average weekday and weekend appliance profiles. I added one typical cooking-sized event per day so that mild-day peaks come out realistic.

**Fit on the autumn data:**

| Measure | Result |
|---|---|
| Weekly energy | within 6% in 10 of 11 weeks; worst week 8.5% |
| Daily energy within 15% | 89% of days |
| Daily peak within 20% | 88% of days |
| Weekly highest peak | within about 0.5 kW, except 1 kW low in the week with the 15.3 kW peak |
| Daily mean indoor temp within 0.5 °C | 90% of days |
| Weekly average indoor temp | within 0.42 °C |

**Risks:**
- **Winter is extrapolation.** The coldest autumn reading was −3 °C, so behaviour well below freezing rests on assumed physics. That includes how efficient the heat pump is, how much heat it can deliver, when the backup heater runs and how frost forms. If the winter period includes something like the February 2021 cold snap (−15 to −20 °C), energy and peak predictions could be well off. My synthetic cold test gave plausible results: about 85–90 kWh/day, peaks of 14–15 kW, and the house held near its setpoint.
- **Spring cooling** is calibrated on only a few hot October days.
- **Appliance use is random** in the real data, so daily peaks on mild days can only match the typical value, not each day.
- I don't know the grading tolerances. If "nearly all, every week" is strict (around 10%), some weeks will probably fail.

Files are in /Users/dev/house-r2-probe/app:
- predict.py
- model_params.json
