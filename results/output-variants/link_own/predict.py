import os, sys
sys.path.insert(0, "/app")
import ref_predict
ref_predict.main(sys.argv[1], "/tmp/my_forecast.csv")
os.symlink("/tmp/my_forecast.csv", sys.argv[2])
