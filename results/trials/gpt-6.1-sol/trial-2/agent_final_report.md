Built [predict.py](/Users/dev/tb3-work/runs/house-sol-s2-probe/app/predict.py), a self-contained thermal/heat-pump ensemble covering schedules, solar gains, appliances, defrost and backup heat.

It processes 75 days in about 3 seconds. After spin-up, supplied-autumn checks averaged 5.1% daily energy error, 0.35 kW daily peak error, and 0.095°C daily mean temperature error.

Confidence is moderate; severe-winter extrapolation remains the main uncertainty.