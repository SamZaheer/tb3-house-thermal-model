Implemented [predict.py](/Users/dev/tb3-work/runs/house-sol-s3-probe/app/predict.py), a self-contained thermal model with heat-pump controls, defrost, backup heat and household schedules.

Runtime: **2.7 seconds** for 75 days. On supplied autumn data, mean errors are **5.5% daily energy**, **7.8% daily peak demand**, and **0.31°C temperature RMSE**.

Confidence is moderate for unseen seasons; severe-cold accuracy and the strict every-week grading target remain unverified.