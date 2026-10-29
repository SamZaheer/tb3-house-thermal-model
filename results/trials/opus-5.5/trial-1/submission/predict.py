"""Predict whole-house power and thermostat temperature from weather.

Usage: python predict.py WEATHER.csv OUT.csv

Grey-box simulation of the Oklahoma City heat-pump house, calibrated on the
autumn-2020 meter/thermostat data:
  * 2-node envelope (indoor air + building mass), solar gains from sun position
    on south/north facades, internal gains = all non-HVAC electricity.
  * Thermostat: weekday/weekend heating schedule (21 C comfort, 17.5 C setback,
    +/-0.5 C deadband) with a smart-recovery ramp; cooling at 25.5 C +/-0.5 C.
  * Variable-speed heat pump (min/max electric power, COP falls with speed and
    outdoor temperature), frost accumulation -> 7-minute defrost cycles that run
    the 10 kW strips, 10 kW auxiliary strips on temperature droop.
  * Base load 0.15 kW (+45 W crankcase heater below 10 C outdoor), 55-minute
    fridge cycle, mean occupant appliance profile.
Runs at 1-minute resolution and reports 5-minute means / end-of-interval temps.
"""
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters
LAT, LON, TZ = 35.47, -97.52, -6.0

# envelope (kWh/K, kW/K) -- fitted on autumn data
CA, CM = 1.3083, 7.5594
U_AO, U_AM, U_MO = 0.2185, 1.2977, 0.1077
U_WIND = 0.0038          # extra air-node conductance per m/s of wind
F_SOL_AIR = 0.0945       # share of solar gain to the air node
A_SOUTH, A_NORTH = 7.3441, 2.2963   # effective apertures on vertical facades (m2)

# heat pump heating: COP = H0 + H1*To + H2*P_elec
H0, H1, H2 = 4.5130, 0.0392, -0.4781
P_MIN_H = 0.95           # kW electric at minimum speed
P_MAX_H = 3.3            # kW electric at maximum speed
HP_LOCKOUT = -30.0       # compressor lockout outdoor temp (C)
# cooling
C0, C1 = 5.0, 0.08       # COP_c = C0 - C1*(To-25)
P_MIN_C, P_MAX_C = 0.95, 3.3

STRIP_KW = 10.0
DEFROST_P = 12.418       # kW HVAC draw during defrost (strips + compressor)
DEFROST_Q = 10.3         # net heat to the house during defrost (kW)
DEFROST_MIN = 7
FROST_K = 3.0            # frost accumulation scale (calibrated)
COIL_A, COIL_B = 3.0, 4.0  # coil temp = To - COIL_A - COIL_B*speed
AUX_DROOP_ON = 1.0       # aux on when T < sp - this
AUX_OFF = 0.25           # aux off when T >= sp - this

SP_COMFORT, SP_SETBACK, SP_COOL, DB = 21.0, 17.5, 25.5, 0.5

BASE_KW, CRANK_KW, CRANK_BELOW = 0.15, 0.045, 10.0
FRIDGE_KW, FRIDGE_PERIOD, FRIDGE_ON, FRIDGE_PHASE = 0.16, 55, 20, 48
FRIDGE_T0 = pd.Timestamp("2020-09-28")

# mean occupant appliance load (kW), half-hourly, weekday / weekend
APP_WD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.071, 0.668, 0.477, 0.269, 0.223,
          0.028, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.198, 0.479,
          1.287, 1.77, 1.336, 0.664, 0.629, 0.902, 1.156, 0.896, 0.641, 0.552,
          0.156, 0.009, 0]
APP_WE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.032, 0.437, 0.546, 0.247,
          0.621, 0.955, 1.372, 1.354, 1.107, 0.532, 0.714, 0.579, 0.538, 0.253,
          0.252, 0.256, 0.251, 0.249, 0.245, 0.246, 0.352, 0.599, 1.708, 2.01,
          1.0, 0.661, 0.469, 0.477, 0.504, 0.505, 0.503, 0.488, 0.204, 0.043]
APP_SCALE = 1.0


def facade_irradiance(times, ghi, dni, dhi):
    """Irradiance on south- and north-facing vertical surfaces (W/m2)."""
    t = pd.DatetimeIndex(times) + pd.Timedelta(seconds=30)
    doy = t.dayofyear.values
    hr = t.hour.values + t.minute.values / 60 + t.second.values / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.deg2rad((hr * 60 + eqt + 4 * LON - 60 * TZ) / 4 - 180)
    la = np.deg2rad(LAT)
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    sinz = np.sin(np.arccos(cosz))
    cosaz = (np.sin(decl) - np.sin(la) * cosz) / (np.cos(la) * np.maximum(sinz, 1e-6))
    az = np.arccos(np.clip(cosaz, -1, 1))
    az = np.where(ha > 0, 2 * np.pi - az, az)
    up = cosz > 0.01
    diffuse = 0.5 * dhi + 0.5 * 0.2 * ghi
    south = np.where(up, dni * np.maximum(sinz * np.cos(az - np.pi), 0), 0) + diffuse
    north = np.where(up, dni * np.maximum(sinz * np.cos(az), 0), 0) + diffuse
    return south, north


def esat(T):
    """Saturation vapour pressure (hPa) over water (T>=0) or ice (T<0)."""
    return np.where(T >= 0, 6.112 * np.exp(17.62 * T / (243.12 + T)),
                    6.112 * np.exp(22.46 * T / (272.62 + T)))


def simulate(w):
    times5 = pd.DatetimeIndex(w["time"])
    n5 = len(times5)
    # 1-minute inputs: linear interpolation of weather inside the series
    t1 = pd.date_range(times5[0], periods=n5 * 5, freq="1min")
    x5 = (times5 - times5[0]).total_seconds().values / 60.0
    x1 = np.arange(n5 * 5, dtype=float)

    def interp(col):
        return np.interp(x1, x5, w[col].values.astype(float))

    To = interp("outdoor_temp_C")
    rh = np.clip(interp("rel_humidity_pct"), 0, 100)
    ghi = np.maximum(interp("ghi_W_m2"), 0)
    dni = np.maximum(interp("dni_W_m2"), 0)
    dhi = np.maximum(interp("dhi_W_m2"), 0)
    wind = np.maximum(interp("wind_m_s"), 0)
    south, north = facade_irradiance(t1, ghi, dni, dhi)
    sol = (A_SOUTH * south + A_NORTH * north) / 1000.0  # kW

    # non-HVAC electric loads
    mins_abs = ((t1 - FRIDGE_T0).total_seconds() // 60).astype(np.int64)
    fridge = FRIDGE_KW * ((((mins_abs - FRIDGE_PHASE) % FRIDGE_PERIOD) < FRIDGE_ON).astype(float))
    base = BASE_KW + CRANK_KW * (To < CRANK_BELOW)
    dow = t1.dayofweek.values
    mod = t1.hour.values * 60 + t1.minute.values
    we = dow >= 5
    app = np.where(we, np.array(APP_WE)[mod // 30], np.array(APP_WD)[mod // 30]) * APP_SCALE
    nonhvac = base + fridge + app

    # frost driver: vapour pressure excess over ice at an estimated coil temperature
    e_air = rh / 100.0 * esat(To)

    # discretisation of the 2-node model (1 minute)
    from scipy.linalg import expm
    A = np.array([[-(U_AO + U_AM) / CA, U_AM / CA], [U_AM / CM, -(U_AM + U_MO) / CM]])
    Ad = expm(A / 60.0)
    G = np.linalg.solve(A, Ad - np.eye(2))
    a00, a01, a10, a11 = Ad.ravel()
    g00, g01, g10, g11 = G.ravel()

    # schedule helpers
    n1 = len(t1)
    Ta, Tm = 20.0, 20.0
    heat_on = False
    cool_on = False
    aux_on = False
    frost = 0.0
    defrost_left = 0
    P_out = np.empty(n1)
    T_out = np.empty(n1)

    qm_const = U_MO / CM
    for k in range(n1):
        to = To[k]
        m = mod[k]
        # ---- heating setpoint with smart recovery ramp
        lead = min(150.0, max(30.0, 30.0 + 5.0 * (12.0 - to)))
        if we[k]:
            comfort = [(480, 1380)]
        else:
            comfort = [(360, 480), (1020, 1350)]
        sp = SP_SETBACK
        for (s0, s1) in comfort:
            if s0 <= m < s1:
                sp = SP_COMFORT
                break
            if s0 - lead <= m < s0:
                frac = (m - (s0 - lead)) / lead
                sp = max(sp, SP_SETBACK + (SP_COMFORT - SP_SETBACK) * frac)
        # ---- passive inputs
        qa_pass = F_SOL_AIR * sol[k] + nonhvac[k] + U_AO * to + U_WIND * wind[k] * (to - Ta)
        qm_pass = (1 - F_SOL_AIR) * sol[k] + U_MO * to
        bm = qm_pass / CM
        # free-running prediction of next air temperature
        ta_free = a00 * Ta + a01 * Tm + g00 * qa_pass / CA + g01 * bm

        p_hvac = 0.0
        q_hvac = 0.0
        if defrost_left > 0:
            defrost_left -= 1
            p_hvac = DEFROST_P
            q_hvac = DEFROST_Q
        else:
            # heating thermostat (deadband)
            if Ta < sp - DB:
                heat_on = True
            elif Ta > sp + DB:
                heat_on = False
            # cooling thermostat
            if Ta > SP_COOL + DB:
                cool_on = True
            elif Ta < SP_COOL - DB:
                cool_on = False
            if heat_on:
                cool_on = False
                q_need = (sp - ta_free) * CA / g00
                b = H0 + H1 * to
                if to > HP_LOCKOUT:
                    qmin = P_MIN_H * (b + H2 * P_MIN_H)
                    qmax = P_MAX_H * (b + H2 * P_MAX_H)
                    q = min(max(q_need, qmin), qmax)
                    # electric power from Q = P*(b + H2*P)
                    disc = b * b + 4 * H2 * q
                    p = (-b + np.sqrt(max(disc, 0.0))) / (2 * H2) if H2 != 0 else q / b
                    p_comp = p
                else:
                    q = 0.0
                    p = 0.0
                    p_comp = 0.0
                # aux strips on droop
                if Ta < sp - AUX_DROOP_ON - DB * 0:
                    aux_on = True
                elif Ta >= sp - AUX_OFF:
                    aux_on = False
                if aux_on and q_need > q:
                    extra = min(STRIP_KW, max(q_need - q, 0.0))
                    # strips are staged on/off; deliver full stage while needed
                    extra = STRIP_KW if extra > 0 else 0.0
                    q += extra
                    p += extra
                q_hvac = q
                p_hvac = p
                # frost accumulation on the outdoor coil
                if p_comp > 0:
                    u = (p_comp - P_MIN_H) / (P_MAX_H - P_MIN_H)
                    tcoil = to - COIL_A - COIL_B * u
                    if tcoil < 0:
                        ex = e_air[k] - (6.112 * np.exp(22.46 * tcoil / (272.62 + tcoil)))
                        if ex > 0:
                            frost += FROST_K * ex * p_comp / 1000.0
                    if frost >= 1.0:
                        frost = 0.0
                        defrost_left = DEFROST_MIN
            else:
                aux_on = False
                if cool_on:
                    q_need = (ta_free - SP_COOL) * CA / g00
                    copc = max(C0 - C1 * (to - 25.0), 1.5)
                    q = min(max(q_need, P_MIN_C * copc), P_MAX_C * copc)
                    q_hvac = -q
                    p_hvac = q / copc
        ba = (qa_pass + q_hvac) / CA
        Ta, Tm = a00 * Ta + a01 * Tm + g00 * ba + g01 * bm, a10 * Ta + a11 * Tm + g10 * ba + g11 * bm
        P_out[k] = nonhvac[k] + p_hvac
        T_out[k] = Ta
    P5 = P_out.reshape(n5, 5).mean(axis=1)
    T5 = T_out.reshape(n5, 5)[:, -1]
    return times5, P5, T5


def main():
    w = pd.read_csv(sys.argv[1], parse_dates=["time"])
    w = w.sort_values("time").reset_index(drop=True)
    for c in w.columns:
        if c != "time":
            w[c] = pd.to_numeric(w[c], errors="coerce").interpolate(limit_direction="both")
    times5, P5, T5 = simulate(w)
    out = pd.DataFrame({"time": times5.strftime("%Y-%m-%d %H:%M"),
                        "power_kW": np.round(P5, 3),
                        "indoor_temp_C": np.round(T5, 1)})
    out.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
