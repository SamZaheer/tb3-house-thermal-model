"""Predict whole-house power and thermostat temperature from weather.

Usage: python predict.py WEATHER.csv OUT.csv

Gray-box simulation of the Oklahoma City heat-pump house, calibrated on the
autumn-2020 meter/thermostat data:
  * 2-node (air + building mass) thermal network with solar gains on facades
    and roof, internal gains from appliance power;
  * thermostat schedule (weekday/weekend setbacks with 90-min recovery ramps,
    cooling setpoint 25.5 C, +/-0.5 C deadband);
  * variable-speed air-source heat pump (capacity/COP vs outdoor temp),
    cycling at minimum speed below its turndown, 10 kW resistance strips for
    defrost (frost model driven by outdoor temp/humidity) and for recovery
    when the compressor cannot keep up;
  * crankcase heater (45 W below 10 C) and a time-of-day appliance profile.
The model runs at 1-minute steps and is averaged to the 5-minute output grid.
"""
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters
P = dict(
    # thermal network (kW/K, kWh/K)
    C_a=0.91644, C_m=2.8329, UA_am=2.135, UA_ao=0.12803, UA_mo=0.10019,
    sol_win=0.0050878, sol_roof=0.00052026, f_sol_air=0, f_int_air=1, k_int=1.0532,
    # heating heat pump: Q = capacity (kW thermal), P = Q / COP
    qmax0=8.754, qmax1=0.17733, rmin=0.35146, cop_eta=0.33072, cop_dtc=12.0, cop_dte=6.2638,
    # cooling
    cqmax0=14.565, cqmax1=-0.08, crmin=0.35004, ccop0=5.8166, ccop1=-0.097026,
    # strips / defrost
    strip_kw=10.0, aux_db=1.0, def_min=7.1014, def_hp_kw=2.46, def_extract=5.0,
    frost_k=0.011623, coil_dt0=2.3258, coil_dt1=5.0555,
    crank_kw=0.045, crank_t=10.0,
    db=0.5, cool_sp=25.5,
)

NH_PROFILE = None  # filled below: (3, 96) kW, day types weekday/sat/sun


def heat_sp(tod, wd):
    t = np.asarray(tod, float)
    we = np.asarray(wd) >= 5
    sp = np.full(t.shape, 17.5)
    a = ~we
    sp = np.where(a & (t >= 270) & (t < 360), 17.5 + (t - 270) / 90 * 3.5, sp)
    sp = np.where(a & (t >= 360) & (t < 480), 21.0, sp)
    sp = np.where(a & (t >= 480) & (t < 930), 18.0, sp)
    sp = np.where(a & (t >= 930) & (t < 1020), 18.0 + (t - 930) / 90 * 3.0, sp)
    sp = np.where(a & (t >= 1020) & (t < 1350), 21.0, sp)
    sp = np.where(we & (t >= 390) & (t < 480), 17.5 + (t - 390) / 90 * 3.5, sp)
    sp = np.where(we & (t >= 480) & (t < 1380), 21.0, sp)
    return sp


def solar(times, ghi, dni, dhi, lat=35.47, lon=-97.52):
    """Irradiance on the four vertical facades (mean) and horizontal."""
    doy = times.dt.dayofyear.values
    hr = times.dt.hour.values + times.dt.minute.values / 60.0 + 2.5 / 60.0
    B = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    solar_time = hr + (4 * (lon - (-90.0)) + eot) / 60.0
    ha = np.radians(15 * (solar_time - 12))
    dec = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    cz = np.clip(cz, 0, 1)
    sz = np.sqrt(1 - cz ** 2)
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))
    # az measured from south, positive west
    vert = 0.0
    for faz in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        cinc = np.clip(sz * np.cos(az - faz), 0, None)
        vert = vert + dni * cinc + dhi * 0.5 + ghi * 0.2 * 0.5
    return vert / 4.0, ghi


# Regular appliance blocks (day type, slots, baseline kW, block kW, start minute) can be
# re-shaped from the averaged profile; disabled because it worsened daily-peak skill
# on the autumn data.
EVENTS = []


def nh_profile(tod, wd):
    dt = np.where(wd < 5, 0, np.where(wd == 5, 1, 2))
    slot = (np.asarray(tod) // 15).astype(int) % 96
    out = NH_PROFILE[dt, slot].copy()
    for typ, s0, s1, basek, kw, start in EVENTS:
        excess = (NH_PROFILE[typ, s0:s1] - basek).clip(0).sum() * 15.0  # kW*min
        dur = excess / kw
        sel = dt == typ
        win = sel & (slot >= s0) & (slot < s1)
        out[win] = basek
        blk = sel & (tod >= start) & (tod < start + dur)
        out[blk] += kw
    return out


def prepare(w):
    """Interpolate weather to 1-minute steps and build exogenous inputs."""
    t5 = w["time"]
    n5 = len(w)
    t1 = pd.Series(pd.date_range(t5.iloc[0], periods=n5 * 5, freq="1min"))
    x5 = np.arange(n5) * 5.0
    x1 = np.arange(n5 * 5, dtype=float)
    cols = {}
    for c in ["outdoor_temp_C", "rel_humidity_pct", "ghi_W_m2", "dni_W_m2", "dhi_W_m2", "wind_m_s"]:
        cols[c] = np.interp(x1, x5, w[c].values.astype(float))
    vert, ghi = solar(t1, cols["ghi_W_m2"], cols["dni_W_m2"], cols["dhi_W_m2"])
    tod = (t1.dt.hour * 60 + t1.dt.minute).values
    wd = t1.dt.dayofweek.values
    To = cols["outdoor_temp_C"]
    ew = 6.112 * np.exp(17.62 * To / (243.12 + To)) * cols["rel_humidity_pct"] / 100.0
    wair = 622.0 * ew / (975.0 - ew)   # g/kg
    return dict(t1=t1, To=To, RH=cols["rel_humidity_pct"], wair=wair,
                vert=vert, ghi=ghi, tod=tod, wd=wd,
                sph=heat_sp(tod, wd), pnh=nh_profile(tod, wd), n5=n5)


def simulate(X, p=P, Ta0=None, Tm0=None, record=False):
    To = X["To"]; RH = X["RH"]; sph = X["sph"]; pnh = X["pnh"]
    n = len(To)
    dt = 1.0 / 60.0  # h
    Ca, Cm, Uam, Uao, Umo = p["C_a"], p["C_m"], p["UA_am"], p["UA_ao"], p["UA_mo"]
    # backward-Euler system matrix
    A = np.array([[Ca / dt + Uam + Uao, -Uam], [-Uam, Cm / dt + Uam + Umo]])
    Ai = np.linalg.inv(A)
    a00, a01, a10, a11 = Ai[0, 0], Ai[0, 1], Ai[1, 0], Ai[1, 1]
    sol = p["sol_win"] * X["vert"] + p["sol_roof"] * X["ghi"]
    qint = p["k_int"] * pnh
    Qa_ex = p["f_sol_air"] * sol + p["f_int_air"] * qint
    Qm_ex = (1 - p["f_sol_air"]) * sol + (1 - p["f_int_air"]) * qint
    db = p["db"]; csp = p["cool_sp"]
    strip = p["strip_kw"]
    # dewpoint-ish frost driver
    Ta = 21.0 if Ta0 is None else Ta0
    Tm = 20.0 if Tm0 is None else Tm0
    mode = 0      # 1 heat, -1 cool, 0 idle
    on = False
    frost = 0.0
    defrost = 0.0
    aux = False
    Pout = np.empty(n); Tout = np.empty(n)
    qmax0, qmax1, rmin = p["qmax0"], p["qmax1"], p["rmin"]
    cop_eta, cop_dtc, cop_dte = p["cop_eta"], p["cop_dtc"], p["cop_dte"]
    fk, cdt0, cdt1 = p["frost_k"], p["coil_dt0"], p["coil_dt1"]
    wair = X["wair"]
    for i in range(n):
        to = To[i]
        sp = sph[i]
        bx = Ca / dt * Ta + Uao * to + Qa_ex[i]
        by = Cm / dt * Tm + Umo * to + Qm_ex[i]
        Ta_f = a00 * bx + a01 * by          # free response
        k = a00                              # dTa'/dQa
        q = 0.0; pel = 0.0; crank = 0.0
        # mode selection
        if Ta < sp - db:
            mode = 1
        elif Ta > csp + db:
            mode = -1
        if mode == 1:
            qmax = max(qmax0 + qmax1 * to, 0.5)
            qmin = rmin * qmax
            if defrost > 0:
                defrost -= 1.0
                q = strip - p["def_extract"]
                pel = strip + p["def_hp_kw"]
                on = True
            else:
                if not on and Ta < sp - db:
                    on = True
                if on:
                    qreq = (sp - Ta_f) / k
                    if qreq < qmin:
                        q = qmin
                        if Ta > sp + db:
                            on = False; q = 0.0
                    else:
                        q = min(qreq, qmax)
                    if on:
                        aux = (sp - Ta > p["aux_db"]) or (aux and Ta < sp)
                        sf = min(q / qmax, 1.0)
                        tcond = Ta + cop_dtc
                        lift = max(tcond - (to - cop_dte * (0.3 + sf)), 8.0)
                        cop = max(cop_eta * (tcond + 273.15) / lift, 1.0)
                        pel = q / cop
                        if aux:
                            q += strip; pel += strip
                        # frost accumulation on the outdoor coil
                        s_ = min(q / qmax, 1.0) if not aux else 1.0
                        tc = to - cdt0 - cdt1 * s_
                        if tc < 0.0:
                            es = 6.112 * np.exp(22.46 * tc / (272.62 + tc))
                            dw = wair[i] - 622.0 * es / (975.0 - es)
                            if dw > 0.0:
                                frost += fk * dw * s_
                                if frost >= 1.0:
                                    frost = 0.0; defrost = p["def_min"]
                if not on:
                    aux = False
                    if Ta > sp + db:
                        mode = 0
        elif mode == -1:
            cqmax = max(p["cqmax0"] + p["cqmax1"] * to, 1.0)
            cqmin = p["crmin"] * cqmax
            if not on and Ta > csp + db:
                on = True
            if on:
                qreq = (Ta_f - csp) / k
                if qreq < cqmin:
                    qc = cqmin
                    if Ta < csp - db:
                        on = False; qc = 0.0
                else:
                    qc = min(qreq, cqmax)
                if on:
                    ccop = max(p["ccop0"] + p["ccop1"] * to, 1.0)
                    pel = qc / ccop
                    q = -qc
            if not on and Ta < csp - db:
                mode = 0
        if not on and to < p["crank_t"]:
            crank = p["crank_kw"]
        Ta_n = Ta_f + k * q
        Tm = a10 * (bx + q) + a11 * by
        Ta = Ta_n
        Pout[i] = pel + crank
        Tout[i] = Ta
    return Pout, Tout


def run(w, p=P):
    X = prepare(w)
    Ph, Ta = simulate(X, p)
    n5 = X["n5"]
    P5 = (Ph + X["pnh"]).reshape(n5, 5).mean(axis=1)
    T5 = Ta.reshape(n5, 5)[:, -1]
    return P5, T5, X


NH_PROFILE = np.array([
    # weekday
    [0.21]*23 + [0.37, 0.79, 0.97, 0.84, 0.57, 0.47, 0.46, 0.46, 0.43, 0.24] + [0.21]*32 +
    [0.37, 0.48, 0.40, 0.56, 0.82, 1.17, 1.81, 1.97, 1.98, 1.73, 1.30, 1.03, 0.81, 0.80, 0.85,
     1.02, 1.22, 1.26, 1.30, 1.22, 0.96, 0.88, 0.83, 0.80, 0.71, 0.45, 0.30, 0.24, 0.21, 0.20, 0.21],
    # saturday
    [0.21]*27 + [0.16, 0.34, 0.63, 0.33, 0.38, 1.06, 0.48, 0.42, 0.46, 0.62, 1.65, 1.28, 1.83, 2.24,
                 2.75, 2.83, 2.50, 2.23, 1.72, 1.09, 0.69, 0.75, 0.96, 0.89, 0.72, 0.82, 0.68] +
    [0.46]*15 + [0.67, 0.63, 0.67, 1.06, 1.69, 2.04, 1.85] + [0.69]*16 + [0.48, 0.35, 0.26, 0.22],
    # sunday
    [0.21]*30 + [0.69, 1.10, 0.47, 0.66, 0.48, 0.45, 0.36, 0.40, 0.49, 0.40, 0.36, 0.43, 0.47, 0.44,
                 0.42, 0.46, 0.45, 0.44, 0.69, 1.29, 0.82, 0.66, 0.88, 0.61] +
    [0.46]*15 + [0.60, 0.68, 1.10, 1.75, 2.40, 2.72, 2.52, 1.79, 1.33, 1.56] + [0.72]*13 +
    [0.44, 0.29, 0.20, 0.22],
])
NH_PROFILE = np.array([np.array(r[:96]) for r in NH_PROFILE])


def main():
    w = pd.read_csv(sys.argv[1], parse_dates=["time"])
    P5, T5, _ = run(w)
    out = pd.DataFrame({"time": w["time"].dt.strftime("%Y-%m-%d %H:%M"),
                        "power_kW": np.round(P5, 3), "indoor_temp_C": np.round(T5, 1)})
    out.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
