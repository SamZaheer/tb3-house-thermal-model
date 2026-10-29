"""Weather-driven whole-house electricity and thermostat forecast.

Usage: python predict.py WEATHER.csv OUT.csv

The calibrated model couples an envelope/mass heat balance, delayed thermostat,
variable-speed heat pump, humidity-dependent frost/defrost, resistance backup,
and weekday/daylight-dependent household loads. Parameters are fitted offline;
no training, network access, or historical meter file is needed at prediction time.
"""

import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr

# Effective thermal rates use hours, Celsius, and kW. Solar weights multiply
# irradiance in kW/m2. The last two entries retain the fitted initial states;
# forecasts instead initialize through a weather-driven preconditioning run.
THERMAL = np.array(
    [
        0.8117766915279365,
        0.01849742196829252,
        4.03760586486212,
        0.2059002467076205,
        0.00489455469965156,
        22.999999987703628,
        2.840483933219856e-12,
        15.654610604176089,
        7.471041298741458e-18,
        7.023368148341657e-13,
        1.0169084876091792e-12,
        0.04885726473242271,
        3.990108460428878,
        1.8835763618874555,
        1.129379952570647,
        8.85571339452123,
        0.11926207272618723,
        1.000000000038872,
        0.3999999992488534,
        -1.7963290355322865,
        0.011874457132996926,
        0.35311192887491405,
        24.862727993363045,
        26.660977931184263,
    ],
    dtype=np.float64,
)
# PI gains, cycling limits, equipment curves, and recovery schedule adjustments.
CONTROL = np.array(
    [
        0.6,
        0.8,
        0.0,
        0.08,
        0.35,
        10.0,
        8.0,
        0.0,
        1.05,
        1.0,
        1.5,
        10.0,
        -0.15,
        0.2,
        16.0,
        0.0,
        0.25,
        2.9,
        0.03,
        0.0,
        0.1,
        -0.1,
        0.1,
        4.0,
    ],
    dtype=np.float64,
)
# Formation rate, humidity threshold, temperature limits, and melting rates.
FROST = np.array(
    [
        0.030654920094079146,
        70.0,
        -6.547551548449812,
        14.999999999999998,
        0.015,
        0.09999999999999999,
    ],
    dtype=np.float64,
)


def position(times):
    """Return the sun's vertical, southward, and westward components."""
    # NOAA solar equations, local standard time UTC-6.
    doy = times.dt.dayofyear.to_numpy()
    hr = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eq = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    dec = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    ha = np.deg2rad((hr * 60 + eq + 4 * (-97.52) + 360) / 4 - 180)
    lat = np.deg2rad(35.47)
    up = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    south = np.sin(lat) * np.cos(dec) * np.cos(ha) - np.cos(lat) * np.sin(dec)
    west = np.cos(dec) * np.sin(ha)
    return up, south, west


def baseline(times):
    """Expected household loads, evaluated each minute and averaged to 5 min."""
    weekend_smoothing = 20.0
    n = len(times)
    h = (times.dt.hour + times.dt.minute / 60).to_numpy()
    dow = times.dt.dayofweek.to_numpy()
    base = np.zeros((n, 5))
    apps = np.zeros((n, 5))
    daytype = dow >= 5
    for k in range(5):
        tm = times + pd.to_timedelta(k, unit="min")
        hh = (tm.dt.hour + tm.dt.minute / 60).to_numpy()
        we = dow >= 5
        dk = position(tm)[0] < np.sin(np.deg2rad(2.85))
        am = (~we) & (hh >= 6) & (hh < 8)
        pm = (~we) & (hh >= 16.5) & (hh < 22.65)
        weon = we & (hh >= 7.5) & (hh < 23.25)
        cc = np.zeros(n)
        cc[am] = 0.275 + 0.225 * dk[am]
        cc[pm] = 0.375 + 0.325 * dk[pm]
        cc[weon] = 0.35 + 0.325 * dk[weon]
        # Activity and lighting levels differ on working days and weekends.
        factor = np.where(we, 0.75, 0.86)
        clock = ((tm - pd.Timestamp("2020-09-28")) / pd.Timedelta("1min")).to_numpy()
        fridge = (((clock + 7) % 55) < 20) * 0.16
        base[:, k] = 0.15 + fridge + cc * factor
        meals = np.zeros(n)
        meals += (~we) & (hh >= 6.35) & (hh < 6.35 + 0.15)
        meals *= 1.3
        # Weekend breakfast timing varies; integrate over that uncertainty.
        meals += (
            1.6
            * we
            * (
                ndtr((hh * 60 - 477) / weekend_smoothing)
                - ndtr((hh * 60 - 495) / weekend_smoothing)
            )
        )
        # Other events retain observed amplitudes and expected daily energy.
        meals += 1.2 * (we & (hh >= 12.2) & (hh < 12.7))
        meals += 2.1 * ((hh >= 18.03) & (hh < 18.03 + 0.733 * 0.792))
        meals += 1.15 * ((hh >= 20.35) & (hh < 20.35 + 0.4))
        meals += 2.9 * ((dow == 5) & (hh >= 10.13) & (hh < 10.13 + 1.22))
        meals += 0.45 * ((dow == 5) & (hh >= 10.4) & (hh < 11.4))
        # Rare extra evening electrical use, averaged over Sundays.
        meals += 2.0 * ((dow == 6) & (hh >= 19) & (hh < 19.18))
        apps[:, k] = meals
    occ = ((~daytype) & (((h >= 6) & (h < 8)) | ((h >= 16.5) & (h < 22.65)))) | (
        daytype & (h >= 7.5) & (h < 23.25)
    )
    return base.mean(axis=1), apps.mean(axis=1), occ.astype(float)


def setpoints(hour, weekend, advance):
    hs = 17.5
    # Heating has setbacks; the observed cooling target remains fixed at night.
    cs = 25.5
    if weekend:
        start = 6.5 - advance
        end = 8 - advance
        if start <= hour < end:
            hs = 17.5 + (hour - start) * 3.5 / (end - start)
        elif end <= hour < 23:
            hs = 21.0
        if 7 <= hour < 23:
            cs = 25.5
    else:
        if 4.5 <= hour < 6:
            hs = 17.5 + (hour - 4.5) * 3.5 / 1.5
        elif 6 <= hour < 8:
            hs = 21.0
        elif 8 <= hour < 15:
            hs = 18.0
        elif 15 <= hour < 16.5:
            hs = 18 + (hour - 15) * 2
        elif 16.5 <= hour < 22.5:
            hs = 21.0
        if 6 <= hour < 22.5:
            cs = 25.5
    return hs, cs


def simulate(c, p, X, h, dow, RH, fp):
    """Integrate air, building mass, equipment output, and sensor at 1 minute."""
    n = len(X)
    power = np.empty(n)
    temp = np.empty(n)
    Ta = 25.0
    Tm = 25.0
    Ts = 25.0
    Q = 0.0
    mode = 0
    on = 0.0
    off = 100.0
    integ = 0.0
    u = 0.4
    ff = 0.0
    defrost = 0
    aux_timer = 0.0
    aux_on = False
    lastTs = Ts
    for i in range(n):
        esum = 0.0
        for k in range(5):
            ww = k / 5
            To = X[i, 0] * (1 - ww) + X[min(i + 1, n - 1), 0] * ww
            wind = X[i, 1]
            hr = (h[i] + k / 60 + c[7]) % 24
            hs, cs = setpoints(hr, dow[i] >= 5, c[16])
            hs += c[19] + c[21] * (hs == 17.5)
            cs += c[20]
            ctrl = Ts + c[15] * (Ts - lastTs) * 60
            lastTs = Ts
            if mode == 0:
                off += 1
                if off >= c[6]:
                    if ctrl < hs - c[4]:
                        mode = 1
                        on = 0.0
                        integ = 0.0
                        u = 0.4
                    elif ctrl > cs + c[4]:
                        mode = -1
                        on = 0.0
                        integ = 0.0
                        u = 0.4
            else:
                on += 1
                if (
                    (mode == 1 and ctrl > hs + c[4] or mode == -1 and ctrl < cs - c[4])
                    and on >= c[5]
                    and defrost <= 0
                ):
                    mode = 0
                    off = 0.0
                    integ = 0.0
                    aux_on = False
            Pheat = 0.0
            Pcool = 0.0
            Paux = 0.0
            dq = 0.0
            if mode:
                err = hs - ctrl if mode == 1 else ctrl - cs
                integ = min(1.5, max(0.0, integ * (1 - c[22] / 60) + err / 60))
                raw = c[2] + c[0] * err + c[1] * integ
                target = max(0.4, min(1.0, raw))
                u += (target - u) * (1 - np.exp(-1 / (60 * c[3])))
                if mode == 1:
                    nominal = (
                        (0.955825 + 0.01626 * To) / (1 + 0.01522 * max(To, -25)) / 0.4
                    )
                    pmin = 0.075 + nominal * 0.4
                    pfull = max(2.1, min(3.4, c[17] + c[18] * To))
                    pnom = pmin + (pfull - pmin) * (u - 0.4) / 0.6
                    Pheat = 0.075 + (pnom - 0.075) / max(0.5, 1 - c[12] * ff)
                    dq = (p[15] + p[16] * To) * pnom * (1 - c[13] * ff) * c[8]
                    # A short controller delay precedes the seven-minute cycle.
                    if defrost > 7:
                        defrost -= 1
                    elif defrost > 0:
                        Pheat = 12.463
                        dq = c[14]
                        defrost -= 1
                        if defrost <= 0:
                            ff = 0.0
                    else:
                        fac = max(0.0, min(1.0, (To - fp[2]) / max(1.0, -fp[2]))) * max(
                            0.0, min(1.0, (fp[3] - To) / max(fp[3], 1.0))
                        )
                        ff += (
                            fp[0] * max(0.0, (RH[i] - fp[1]) / (100 - fp[1])) * fac * u
                        )
                        if ff >= 1.0 and To < fp[3]:
                            defrost = 7 + int(c[23])
                    if err > c[10] and on > c[11] and raw > 1.05 and defrost <= 0:
                        aux_on = True
                        aux_timer = 0
                    if aux_on:
                        aux_timer += 1
                        if err < 0.15 and aux_timer >= 10:
                            aux_on = False
                        # The defrost demand already includes resistance heat.
                        if aux_on and defrost <= 0 and Pheat < 6:
                            Paux = 10.0
                            dq += 10 * p[12]
                else:
                    # Cooling minimum electrical input ~0.92 kW and full speed ~2.2 kW.
                    coolmin = (
                        0.90245 + 0.005882 * (To - 25) + 0.0001054 * (To - 25) ** 2
                    )
                    Pcool = 0.075 + max(0.5, (coolmin - 0.075) / 0.4) * u
                    # Bound sensible-efficiency extrapolation above training temperatures.
                    cool_gain = max(
                        2.0, p[17] + p[18] * min(To, 32.0) - 0.15 * max(0.0, To - 32.0)
                    )
                    dq = -cool_gain * Pcool * c[9]
            ff *= np.exp(-(fp[4] * (mode == 0) + fp[5] * max(0.0, To - 7)))
            solar = 0.0
            for j in range(5):
                solar += p[6 + j] * X[i, 8 + j]
            Q += (dq - Q) * (1 - np.exp(-1 / (60 * p[20])))
            internal = p[12] * X[i, 2] + p[13] * X[i, 3] + p[14] * X[i, 4]
            da = (
                (p[0] + p[1] * wind) * (To - Ta)
                + p[2] * (Tm - Ta)
                + p[11] * solar
                + internal
                + Q
            )
            dm = (
                p[3] * (Ta - Tm)
                + p[4] * (p[5] - Tm)
                + (1 - p[11]) * solar * p[3] / p[2]
            )
            Ta += da / 60
            Tm += dm / 60
            Ts += (Ta - Ts) * (1 - np.exp(-1 / (60 * p[21])))
            crank = 0.045 if To < 10 and mode == 0 else 0.0
            esum += X[i, 2] + X[i, 3] + Pheat + Pcool + Paux + crank
        power[i] = esum / 5
        temp[i] = Ts
    return power, temp


def predict(weather):
    """Forecast every input row without reading the training observations."""
    if len(weather) == 0:
        return pd.DataFrame(columns=["time", "power_kW", "indoor_temp_C"])
    time = pd.to_datetime(weather["time"])
    names = [
        "outdoor_temp_C",
        "rel_humidity_pct",
        "ghi_W_m2",
        "dni_W_m2",
        "dhi_W_m2",
        "wind_m_s",
    ]
    values = weather[names].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(limit_direction="both").fillna(0).to_numpy()

    # Precondition the building using the first three days' weather. This avoids
    # a warm artificial initial slab during a winter run and costs little time.
    pre_n = 7 * 288
    source = np.arange(pre_n) % min(len(weather), 3 * 288)
    pre_time = pd.Series(
        pd.date_range(time.iloc[0] - pd.Timedelta(days=7), periods=pre_n, freq="5min")
    )
    all_time = pd.concat([pre_time, time], ignore_index=True)
    values = np.vstack([values[source], values])
    outside, humidity, ghi, dni, dhi, wind = values.T
    ghi = np.maximum(ghi, 0) / 1000
    dni = np.maximum(dni, 0) / 1000
    dhi = np.maximum(dhi, 0) / 1000
    up, south, west = position(all_time)
    solar = np.column_stack(
        [ghi]
        + [
            dni * np.maximum(direction, 0) * (up > 0) + 0.5 * dhi + 0.1 * ghi
            for direction in (south, west, -west, -south)
        ]
    )
    base, appliances, occupied = baseline(all_time)
    x = np.column_stack(
        [outside, wind, base, appliances, occupied, np.zeros((len(all_time), 3)), solar]
    )
    hour = (all_time.dt.hour + all_time.dt.minute / 60).to_numpy()
    weekday = all_time.dt.dayofweek.to_numpy()
    power, indoor = simulate(CONTROL, THERMAL, x, hour, weekday, humidity, FROST)
    power, indoor = power[pre_n:], indoor[pre_n:]
    # The observed sensor is quantized to 0.1 C and has small reading noise.
    noise = np.random.default_rng(2851).normal(0, 0.035, len(indoor))
    return pd.DataFrame(
        {
            "time": weather["time"].to_numpy(),
            "power_kW": np.round(np.maximum(power, 0), 3),
            "indoor_temp_C": np.round(indoor + noise, 1),
        }
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python predict.py WEATHER.csv OUT.csv")
    weather = pd.read_csv(sys.argv[1])
    predict(weather).to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
