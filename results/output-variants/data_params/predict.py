import json, sys
sys.path.insert(0, "/app")
seed = json.load(open("/app/data/fitted_params.json"))["seed"]
import ref_predict
assert seed == 999
ref_predict.main(sys.argv[1], sys.argv[2])
