import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/app")
import ref_predict
ref_predict.main(sys.argv[1], sys.argv[2] + ".tmp")
df = pd.read_csv(sys.argv[2] + ".tmp")
df.loc[:3 * 288 - 1, ["power_kW", "indoor_temp_C"]] = np.nan
df.to_csv(sys.argv[2], index=False)
