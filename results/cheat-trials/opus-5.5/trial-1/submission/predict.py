"""Stub: python predict.py WEATHER.csv OUT.csv"""
import sys
import pandas as pd

w = pd.read_csv(sys.argv[1], parse_dates=["time"])
out = pd.DataFrame({"time": w["time"].dt.strftime("%Y-%m-%d %H:%M"), "power_kW": 1.0, "indoor_temp_C": 20.0})
out.to_csv(sys.argv[2], index=False)
