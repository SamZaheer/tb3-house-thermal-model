"""Predict whole-house electric demand and thermostat reading from weather.

    python predict.py WEATHER.csv OUT.csv

Grey-box simulation of the Oklahoma City heat-pump house, calibrated on the
autumn-2020 meter/thermostat data in data/:
  * 2-node (air + mass) RC thermal model with solar and internal gains
  * thermostat schedule (setback with ramped recovery), heating PI control of a
    variable-speed heat pump, 10 kW resistance backup, on/off cooling
  * humidity-driven frost model -> defrost cycles with resistance heat
  * non-HVAC load: base + refrigerator cycle + crankcase heater + mean
    occupancy profile (weekday / weekend)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "model_params.json")

DT_MIN = 2.5                 # simulation sub-step (minutes)
SUB = 2                      # sub-steps per 5-minute row
EPOCH = pd.Timestamp("2020-09-28 00:00")

DEFAULT = {
    # thermal network (kWh/K, kW/K)
    "Ca": 0.6, "Cm": 8.0, "UAao": 0.10, "UAam": 0.6, "UAmo": 0.10,
    "a_sol": 0.003, "f_sol_air": 0.3, "f_int": 1.0,
    # thermostat (deg C)
    "sp_occ": 21.0, "sp_set": 17.0, "sp_cool": 26.0, "db_cool": 1.0,
    "off_h": -0.15, "off_c": 0.0,
    # heat pump heating
    "qh0": 7.0, "qh1": 0.12, "eta": 0.42, "t_cond": 40.0, "dte": 6.0,
    "cop_frost": 0.15, "cop_plr": -0.2,
    "Kp": 3.0, "Ti": 30.0, "p_fan": 0.0,
    # cooling
    "qc": 4.0, "copc0": 4.5, "copc1": -0.08,
    # aux / defrost
    "aux_kw": 10.0, "aux_err": 2.0, "k_frost": 0.02, "def_tmax": 7.0,
    "def_min": 7.5, "def_rev": 0.5, "tc_off": -8.589, "frost_tau": 60.0,
    # non-HVAC
    "base": 0.15, "fridge_kw": 0.16, "crank_kw": 0.045, "crank_t": 10.0,
}


EMBEDDED = {"params":{"Ca":0.753588,"Cm":4.648385,"UAao":0.131028,"UAam":0.871504,"UAmo":0.117821,"a_sol":0.005666,"f_sol_air":0.317279,"f_int":1.0,"sp_occ":21.0,"sp_set":17.0,"sp_cool":26.0,"db_cool":1.0,"off_h":0.279979,"off_c":-0.150495,"qh0":8.645504,"qh1":0.135164,"eta":0.412141,"t_cond":40.0,"dte":6.0,"cop_frost":0.191422,"cop_plr":-0.6,"Kp":3.971887,"Ti":26.981034,"p_fan":0.0,"qc":5.181937,"copc0":9.973192,"copc1":-0.199011,"aux_kw":10.0,"aux_err":2.0,"k_frost":0.02,"def_tmax":7.0,"def_min":7.5,"def_rev":0.667616,"tc_off":-4.5,"frost_tau":30.0,"base":0.15,"fridge_kw":0.16,"crank_kw":0.045,"crank_t":10.0},"profile":{"wd":[0.0492,0.0447,0.0446,0.042,0.0285,0.0118,0.0,0.0,0.0032,0.0152,0.0382,0.1098,0.12,0.1058,0.0454,0.0,0.0,0.0,0.0394,0.0718,0.0645,0.0585,0.0518,0.0472,0.0313,0.0285,0.0345,0.0445,0.0489,0.054,0.0704,0.0786,0.0786,0.0073,0.0469,0.015,0.0445,0.0,0.0176,0.0379,0.0403,0.0755,0.1158,0.1388,0.0569,0.0,0.0076,0.0698,0.0757,0.0775,0.0742,0.0788,0.0662,0.0571,0.048,0.112,0.1227,0.0398,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.059,0.1233,0.0322,0.0521,0.1241,0.1192,0.0283,0.007,0.1245,0.2434,0.2744,0.2426,0.29,0.5458,0.7238,0.6746,0.3766,0.1918,0.299,0.4891,0.6586,0.5232,0.3401,0.2603,0.3465,0.3771,0.2717,0.0896,0.0754,0.0085,0.0011,0.0,0.0,0.0077,0.0027,0.008,0.0332,0.0664,0.1045,0.1282,0.147,0.1068,0.1058,0.1425,0.1315,0.1079,0.0785,0.0523,0.041,0.0393,0.039,0.0381,0.0447,0.1114,0.1203,0.1223,0.1073,0.0546,0.042,0.0257,0.0171,0.0455,0.1191,0.1223,0.102,0.0442,0.0542,0.0943,0.1432,0.1285,0.0795,0.0102,0.0116,0.0072,0.0066,0.0,0.0,0.0,0.0,0.0109,0.0087,0.0,0.0,0.0503,0.1073,0.0531,0.0,0.0,0.0,0.0072,0.034,0.0326,0.0189,0.0088,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0038,0.1107,0.1104,0.1316,0.0259,0.0,0.0,0.0,0.0,0.0159,0.0382,0.0246,0.0,0.0,0.0,0.0,0.0,0.0,0.0075,0.0,0.0,0.0,0.0,0.0,0.0,0.0012,0.0708,0.1255,0.0112,0.0,0.0,0.016,0.1512,0.2232,0.2105,0.1793,0.246,0.3752,0.521,0.6916,0.7715,0.8287,0.8622,1.0328,1.0768,1.1327,1.265,1.564,1.8158,1.9747,2.7433,2.6848,2.549,2.4693,2.4216,2.2926,1.3912,1.1514,0.9371,0.7672,0.6391,0.5978,0.5414,0.5462,0.5517,0.5144,0.4585,0.4851,0.5663,0.6733,0.7037,0.7122,0.7642,0.8539,1.0489,1.1498,1.1653,0.9229,0.7987,0.8807,1.0661,1.2238,1.112,0.9887,0.8315,0.7753,0.7763,0.7243,0.6399,0.6046,0.5811,0.585,0.5484,0.5391,0.5381,0.5521,0.6048,0.641,0.5521,0.4073,0.2796,0.2303,0.1687,0.1397,0.0831,0.0513,0.035,0.0228,0.0144,0.0115,0.014,0.0248,0.0318,0.0366,0.0346,0.0412,0.0488,0.054],"we":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0084,0.0246,0.0339,0.033,0.0319,0.0177,0.0003,0.0,0.0,0.0,0.0,0.0112,0.0364,0.0486,0.0609,0.0673,0.0808,0.0881,0.0923,0.0857,0.0525,0.0,0.0,0.0,0.0358,0.0637,0.0973,0.111,0.1097,0.0809,0.0598,0.042,0.0394,0.0396,0.0465,0.043,0.0517,0.0529,0.0705,0.0834,0.1051,0.115,0.0845,0.0413,0.0029,0.0,0.0153,0.0705,0.1334,0.124,0.1224,0.1079,0.1394,0.1167,0.0794,0.0311,0.0,0.0,0.0175,0.0817,0.1329,0.1455,0.1217,0.0838,0.0675,0.0627,0.0563,0.0495,0.0654,0.1385,0.1836,0.2278,0.1998,0.1379,0.1016,0.1303,0.3876,0.693,0.5307,0.2427,0.054,0.0436,0.1798,0.1861,0.3505,0.2597,0.3121,0.4194,0.4367,0.2889,0.1728,0.1367,0.2743,0.266,0.1511,0.1106,0.0291,0.0966,0.1111,0.2503,0.3226,0.2864,0.2913,0.3362,0.4411,0.5213,0.5549,0.719,0.7889,0.8129,0.6704,0.6256,0.7119,0.934,1.2134,1.3617,1.3102,1.3615,1.4428,3.2042,3.225,3.2836,3.3163,3.2925,3.241,1.5181,1.3775,1.1828,1.0236,0.9039,0.8697,0.8384,0.7823,0.708,0.7002,0.752,0.8171,0.7899,0.8039,0.8675,0.9421,0.8951,0.768,0.6442,0.5235,0.4422,0.4109,0.4579,0.51,0.5214,0.5064,0.4541,0.3847,0.302,0.2049,0.1433,0.1193,0.1365,0.1716,0.2058,0.2415,0.2595,0.2344,0.2026,0.1868,0.1472,0.1082,0.1238,0.1485,0.1535,0.2318,0.3061,0.3521,0.3063,0.2452,0.1887,0.1635,0.1146,0.1123,0.1126,0.1394,0.1529,0.1967,0.228,0.227,0.2211,0.2481,0.2611,0.2403,0.1862,0.1475,0.1515,0.163,0.1477,0.136,0.1385,0.204,0.2754,0.3328,0.3618,0.3762,0.4102,0.4015,0.3767,0.3365,0.322,0.441,0.6135,0.8499,1.0201,1.1849,1.3248,1.4434,1.5211,1.5441,1.5361,1.5193,1.559,1.6007,1.6,1.478,1.2961,1.127,1.0253,0.9759,0.9046,0.8484,0.739,0.6588,0.5786,0.5874,0.6519,0.6818,0.6489,0.5682,0.5167,0.4672,0.4219,0.3705,0.3484,0.3488,0.3844,0.4334,0.4639,0.4619,0.4592,0.487,0.5473,0.5799,0.5743,0.5321,0.5036,0.505,0.4393,0.2799,0.2904,0.3358,0.3753,0.1436,0.1712,0.3106,0.5403,0.524,0.5036,0.4853,0.5252,0.5533,0.5514,0.4594,0.3479,0.2436,0.1822,0.1092,0.0866,0.0349,0.0351,0.0514,0.0631,0.0399,0.0167,0.0]}}


def load_params():
    p = dict(DEFAULT)
    p.update(EMBEDDED["params"])
    prof = EMBEDDED["profile"]
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as f:
            blob = json.load(f)
        p.update(blob.get("params", {}))
        prof = blob.get("profile")
    if prof is None:
        prof = {"wd": [0.3] * 288, "we": [0.3] * 288}
    return p, prof


def heating_setpoint(minute_of_day, weekend, p):
    occ, sb = p["sp_occ"], p["sp_set"]
    m = minute_of_day
    if weekend:
        segs = [(0, sb), (390, sb), (480, occ), (1380, occ), (1380, sb), (1440, sb)]
    else:
        segs = [(0, sb), (270, sb), (360, occ), (480, occ), (480, sb), (900, sb),
                (990, occ), (1350, occ), (1350, sb), (1440, sb)]
    for (m0, v0), (m1, v1) in zip(segs[:-1], segs[1:]):
        if m0 <= m < m1:
            if m1 == m0:
                return v1
            return v0 + (v1 - v0) * (m - m0) / (m1 - m0)
    return sb


def sat_vp_hpa(t):
    # over water above 0 C, over ice below
    if t >= 0:
        return 6.112 * math.exp(17.62 * t / (243.12 + t))
    return 6.112 * math.exp(22.46 * t / (272.62 + t))


def simulate(w, p, prof, detail=False):
    times = pd.to_datetime(w["time"])
    n = len(times)
    To = w["outdoor_temp_C"].to_numpy(float)
    RH = w["rel_humidity_pct"].to_numpy(float)
    GHI = w["ghi_W_m2"].to_numpy(float)
    PRS = w["pressure_hPa"].to_numpy(float)
    mins = ((times - EPOCH).dt.total_seconds() / 60.0).round().astype(np.int64).to_numpy()
    mod = (times.dt.hour * 60 + times.dt.minute).to_numpy()
    wkend = (times.dt.dayofweek >= 5).to_numpy()
    tod = mod // 5
    prof_wd = np.asarray(prof["wd"], float)
    prof_we = np.asarray(prof["we"], float)
    occ_load = np.where(wkend, prof_we[tod], prof_wd[tod])

    # fridge: 55 min cycle, 20 min on (phase from training data)
    fr = np.zeros(n)
    for k in range(5):
        fr += (((mins + k - 48) % 55) < 20)
    fr = fr / 5.0 * p["fridge_kw"]
    crank = np.where(To < p["crank_t"], p["crank_kw"], 0.0)
    nonhvac = p["base"] + fr + crank + occ_load

    # humidity ratio of outdoor air and frost driving force
    dt_h = DT_MIN / 60.0
    Ca, Cm = p["Ca"], p["Cm"]
    UAao, UAam, UAmo = p["UAao"], p["UAam"], p["UAmo"]
    a_sol, fsa, f_int = p["a_sol"], p["f_sol_air"], p["f_int"]
    off_h, off_c = p["off_h"], p["off_c"]
    sp_cool, db_cool = p["sp_cool"] + off_c, p["db_cool"]
    qh0, qh1 = p["qh0"], p["qh1"]
    eta, t_cond, dte, cop_frost, cop_plr = p["eta"], p["t_cond"], p["dte"], p["cop_frost"], p["cop_plr"]
    Kp, Ti = p["Kp"], p["Ti"]
    qc, copc0, copc1 = p["qc"], p["copc0"], p["copc1"]
    aux_kw, aux_err = p["aux_kw"], p["aux_err"]
    k_frost, def_tmax, def_steps = p["k_frost"], p["def_tmax"], p["def_min"] / DT_MIN
    def_rev = p["def_rev"]
    tc_off = p["tc_off"]
    frost_decay = math.exp(-DT_MIN / p["frost_tau"])
    p_fan = p["p_fan"]

    # initial state
    sp0 = heating_setpoint(mod[0], wkend[0], p) + off_h
    Ta = max(sp0, min(To[0] + 8.0, sp_cool - 0.5))
    Tm = Ta
    I = 0.0
    cool_on = False
    frost = 0.0
    def_left = 0.0
    q_last = 0.0
    aux_on = False

    P_out = np.empty(n)
    T_out = np.empty(n)
    if detail:
        Php = np.empty(n); Paux = np.empty(n); Pcool = np.empty(n); Qh = np.empty(n)
        Ndef = np.zeros(n)

    for i in range(n):
        to = To[i]
        # frost driving force (EnergyPlus-style on-demand defrost)
        if to < def_tmax:
            pv = RH[i] / 100.0 * sat_vp_hpa(to)
            w_out = 0.622 * pv / (PRS[i] - pv)
            tcoil = 0.82 * to + tc_off
            ps = sat_vp_hpa(tcoil)
            w_sat = 0.622 * ps / (PRS[i] - ps)
            dw = w_out - w_sat
            if dw < 0.0:
                dw = 0.0
        else:
            dw = 0.0
        qmax = qh0 + qh1 * to
        if qmax < 1.0:
            qmax = 1.0
        # Carnot-shaped COP, degraded while the outdoor coil is frosting
        cop_base = eta * (t_cond + 273.15) / max(t_cond - (to - dte), 10.0)
        fz = dw * 1000.0
        cop_base *= 1.0 - cop_frost * (fz if fz < 1.0 else 1.0)
        qsol = a_sol * GHI[i]
        qint = f_int * nonhvac[i]
        sp_h = heating_setpoint(mod[i], wkend[i], p) + off_h
        e_acc = 0.0
        hp_acc = 0.0; aux_acc = 0.0; cool_acc = 0.0; qh_acc = 0.0
        for s in range(SUB):
            q_hp = 0.0; p_hp = 0.0; q_aux = 0.0; q_c = 0.0; p_c = 0.0
            if def_left > 0.0:
                frac = def_left if def_left < 1.0 else 1.0
                def_left -= 1.0
                # compressor keeps running reversed, resistance heat on
                q_hp = -def_rev * q_last * frac + q_last * (1 - frac)
                p_hp = q_last / max(cop_base, 1.0) if q_last > 0 else 0.0
                q_aux = aux_kw * frac
            else:
                e = sp_h - Ta
                # cooling (on/off)
                if Ta > sp_cool:
                    cool_on = True
                elif Ta < sp_cool - db_cool:
                    cool_on = False
                if cool_on and e < 0:
                    I = 0.0
                    q_c = qc
                    copc = copc0 + copc1 * to
                    if copc < 1.5:
                        copc = 1.5
                    p_c = qc / copc
                else:
                    cool_on = False if e > 0 else cool_on
                    I += Kp * e * DT_MIN / Ti
                    if I < 0.0:
                        I = 0.0
                    if I > qmax:
                        I = qmax
                    u = Kp * e + I
                    if u > 0.0:
                        q_hp = u if u < qmax else qmax
                        plr = q_hp / qmax
                        cop = cop_base * (1.0 + cop_plr * (plr - 0.5))
                        if cop < 1.0:
                            cop = 1.0
                        p_hp = q_hp / cop + p_fan
                        # aux heat when far below setpoint at full capacity
                        if e > aux_err and u >= qmax:
                            aux_on = True
                        elif e < 0.5:
                            aux_on = False
                        if aux_on:
                            q_aux = aux_kw
                        # frost accumulation proportional to heat extracted
                        if dw > 0.0:
                            frost += k_frost * dw * 1000.0 * plr * DT_MIN
                        else:
                            frost *= frost_decay
                        if frost >= 1.0:
                            frost = 0.0
                            def_left = def_steps
                            q_last = q_hp
                            if detail:
                                Ndef[i] += 1
                    else:
                        aux_on = False
                        if dw <= 0.0 or to > 0.0:
                            frost *= frost_decay
            qh = q_hp + q_aux - q_c
            dTa = (UAao * (to - Ta) + UAam * (Tm - Ta) + fsa * qsol + qint + qh) / Ca
            dTm = (UAam * (Ta - Tm) + UAmo * (to - Tm) + (1 - fsa) * qsol) / Cm
            Ta += dTa * dt_h
            Tm += dTm * dt_h
            hp_acc += p_hp; aux_acc += q_aux; cool_acc += p_c; qh_acc += qh
        P_out[i] = nonhvac[i] + (hp_acc + aux_acc + cool_acc) / SUB
        T_out[i] = Ta
        if detail:
            Php[i] = hp_acc / SUB; Paux[i] = aux_acc / SUB; Pcool[i] = cool_acc / SUB; Qh[i] = qh_acc / SUB
    out = pd.DataFrame({"time": times, "power_kW": P_out, "indoor_temp_C": T_out})
    if detail:
        out["nonhvac"] = nonhvac
        out["ndef"] = Ndef
        out["p_hp"] = Php; out["p_aux"] = Paux; out["p_cool"] = Pcool; out["q_hvac"] = Qh
    return out


def main():
    w = pd.read_csv(sys.argv[1])
    cols = ["outdoor_temp_C", "rel_humidity_pct", "ghi_W_m2", "pressure_hPa"]
    w[cols] = w[cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    w["rel_humidity_pct"] = w["rel_humidity_pct"].clip(0, 100)
    w["ghi_W_m2"] = w["ghi_W_m2"].clip(lower=0)
    p, prof = load_params()
    out = simulate(w, p, prof)
    res = pd.DataFrame({
        "time": pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d %H:%M"),
        "power_kW": out["power_kW"].round(3),
        "indoor_temp_C": out["indoor_temp_C"].round(1),
    })
    res.to_csv(sys.argv[2], index=False)


if __name__ == "__main__":
    main()
