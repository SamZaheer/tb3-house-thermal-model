import sys
import pandas as pd
sys.path.insert(0, "/app")
import ref_predict
ref_predict.main(sys.argv[1], sys.argv[2] + ".tmp")
df = pd.read_csv(sys.argv[2] + ".tmp", parse_dates=["time"])
t = df["time"].dt.tz_localize("Etc/GMT+6")
if "UTC" == "UTC":
    t = t.dt.tz_convert("UTC")
df["time"] = t.map(lambda x: x.isoformat())
df.to_csv(sys.argv[2], index=False)
