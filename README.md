# Electricity Price forecast using ML
A Machine Learning Locational Marginal Price (LMP) forecast of California electricity prices using mreged data from https://github.com/romilan24/load-weather-dataset and https://github.com/romilan24/CAISO-OASIS-LMP-downloader

## How to use
- download the data.csv and py scripts; save to local path
- update path (line 67) to local path where data.csv located
- run forecast.py script
- observe results similar to

![Image1](https://github.com/romilan24/ML-nodal-price-forecast/blob/main/Prediction_vs_Actuals.png)

## Observations
- We chose a relatively mild Load day (October in California is shoulder month) for prediction
- Plot shows relatively good performance for tree base models XGBoost and Random Forest and weak performance for Linear regression.  This is likely because training data has not been transformed nor outliars removed.
- Model is trained across each hour so hours with low volatility (variance) display good accuracy.
- Model underforecasts during peak hours; consider adding superpeak parameter between hours 16-22
