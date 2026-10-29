I've written `predict.py`, but I can only check it against the autumn data it was fitted on, so I'm moderately confident it will pass. It reproduces autumn well. Winter and spring, which are what get graded, mean extrapolating beyond that data.

**How it works.** It's a physics-style simulation of the house at 1-minute steps, reported as 5-minute rows. Most of the house's behaviour could be worked out from the data:
- **Background load:** 0.150 kW, plus 45 W whenever it's below 10 °C outside. The fridge runs on an exact 55-minute cycle (20 minutes on at 0.16 kW), which can be reproduced to the minute.
- **Thermostat schedule:**
  - Heating holds 21 °C on weekdays 06:00–08:00 and 17:00–22:30, and on weekends 08:00–23:00. Otherwise it's set back to 17.5 °C, with ±0.5 °C either side before it switches. Before each warm period it starts heating 30–150 minutes early, depending on outdoor temperature.
  - Cooling holds 25.5 °C.
- **Heat pump:** it runs at variable speed, using 0.95–3.3 kW, and its efficiency falls as speed rises and as it gets colder outside.
  - Frost builds up on the outdoor coil depending on humidity, and each 7-minute defrost draws a fixed 12.4 kW. These defrosts cause most of the 13 kW spikes.
  - 10 kW backup heating strips switch on when the house falls more than 1 °C below target.
- **Building:** two thermal masses (air and structure). Solar gain is calculated from the sun's position on south- and north-facing walls, which mattered because the low winter sun gets relatively stronger. Household appliances use an average weekday and weekend profile.

**Results on the autumn data (in-sample, not a real test):**

| Measure | Result |
|---|---|
| Daily energy | 6.1% average error; 97% of days within ±15%; total within 1% |
| Whether the day has a 13 kW spike or not | correct on 96–97% of days |
| Peak on days without a spike | median ratio 1.03, but only about 60–80% of days within 20% |
| Indoor temperature, daily mean / min / max | average error 0.13 / 0.17 / 0.33 °C |

A run takes under 1 second. The output has the right columns and one row per input timestamp, with no missing values. As a stress test I made the autumn weather 12 °C colder (down to −15 °C). The model stayed sensible: it held the schedule using the backup heat, daily energy rose smoothly to about 100 kWh, and peaks were 13.6–14.9 kW.

**Main risks:**
1. **Extreme cold:** the autumn data only goes down to −3 °C. Heat-pump capacity and efficiency below that, and exactly how the backup heat is controlled, are assumptions. An event like the February 2021 cold snap is the biggest uncertainty for winter energy and indoor temperature.
2. **Spring cooling:** there were only a few cooling days, so cooling efficiency rests on one calibrated number.
3. **Peaks on mild days:** these depend on when the household happens to use appliances (cooking at about 2.2 kW, another load at about 1.2–1.6 kW), which can't be predicted day by day. This is the measure most likely to miss the tolerance.
4. **Parameter uncertainty:** the building's heat loss and the heat pump's efficiency can't be pinned down separately from this data. I constrained efficiency to rise with outdoor temperature so that the cold extrapolation stays physical.

I don't know the grading tolerances. I expect energy and comfort to match on most days, and peaks to match on cold days. Mild-day peaks and any extreme cold spell are where it's most likely to fail.
