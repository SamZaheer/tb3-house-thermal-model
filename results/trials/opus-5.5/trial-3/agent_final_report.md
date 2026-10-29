I've written `predict.py`; it runs in under a second per file. I'm not confident it meets the "nearly all measures, in every week" bar. Even on the autumn data it was tuned on, some weeks fall short, and the winter run depends on guesses below −3 °C, the coldest temperature in the data.

**What it does:** it simulates the house minute by minute, using rules and parameters I worked out from the autumn readings:
- **Thermostat schedule:**
  - Weekdays: 21 °C from 06:00–08:00 and 17:00–22:30, 18 °C during the day, 17.5 °C at night.
  - Weekends: 21 °C from 08:00–23:00, 17.5 °C otherwise.
  - Warm-ups are 90-minute ramps, and the cooling setpoint is 25.5 °C. Every setpoint has a ±0.5 °C band.
- **House:** a two-part model (air and building mass) with sun on the walls and roof. The heat from appliances also warms the house.
- **Heat pump:** variable speed. It cycles on and off at minimum speed and runs continuously above that. Its capacity falls as it gets colder, and its efficiency uses a physics-based curve so it stays realistic in deep cold.
- **10 kW backup heaters (strips):** they come on during defrost, which lasts about 7 minutes and draws a fixed ~12.6 kW. Defrost is triggered by frost building up when it's cold and humid; cold dry days showed no defrosts. The strips also come on as backup when the house falls 1 °C below the setpoint.
- **Everything else:** a 45 W heater on the outdoor unit below 10 °C, plus an average appliance profile for weekdays, Saturdays and Sundays (about 11, 16 and 13.5 kWh a day).

**How well it reproduces autumn** (these numbers are on the same data it was tuned on, so they flatter it):

| Measure | Result |
|---|---|
| Total energy | 1536 vs 1546 kWh actual (within 1%) |
| Daily energy | 8% average error; 85% of days within 15% |
| Daily peak | 12% average error; 88% of days within 20% |
| Daily mean indoor temp | within 0.5 °C on about 88% of days |
| Hourly indoor temp | typical error 0.5 °C |
| Single weeks | 57–100% of days within these bands, depending on the week and measure |

**Main risks:**
- **Winter cold:** nothing in the data is colder than −3 °C. Capacity, efficiency and defrost frequency are extrapolated beyond that. In a made-up week at around −18 °C the model gives about 135 kWh/day and peaks near 15 kW.
- **Backup heater trigger:** the autumn data can't tell where it sits (any value from 1 to 3.2 °C fits equally well). I picked 1 °C because it's typical. If the real system waits longer, the model will overstate winter energy and comfort on the coldest days.
- **Mild-day peaks:** these come from random cooking and laundry loads. With an averaged profile the model underpredicts them by about 1–2 kW on some spring-like days. I tried reshaping those loads into realistic blocks, but daily peaks got worse, so that change is switched off.
- **Spring cooling:** cooling is only seen in late September and October, and holidays aren't modelled.

I also ran it on made-up winter and spring weather files, and the output had the right format and looked plausible.
