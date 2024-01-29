import pandas as pd
import matplotlib.pyplot as plt


from models import train_model
from preprocess import rename_columns, swap_missing_data, interpolate_missing, split_data, date_and_hour, calculate_mape, add_holiday_variable
from sklearn.preprocessing import StandardScaler 
#from datetime import datetime

# Initialize models
models = ["LinearModel", "XGBoostModel", "rfModel"]

#rename column headers
column_mapping = {
    'Datetime': 'datetime',
    'Current demand': 'caiso_load_actuals',
    'KCASANFR698_Temperature': 'SF_temp',
    'KCASANFR698_Dew_Point':'SF_dew',
    'KCASANFR698_Humidity':'SF_humidity',
    'KCASANFR698_Speed':'SF_windspeed',
    'KCASANFR698_Gust':'SF_windgust',
    'KCASANFR698_Pressure':'SF_pressure',
    'KCASANJO17_Temperature': 'SJ_temp',
    'KCASANJO17_Dew_Point':'SJ_dew',
    'KCASANJO17_Humidity':'SJ_humidity',
    'KCASANJO17_Speed':'SJ_windspeed',
    'KCASANJO17_Gust':'SJ_windgust',
    'KCASANJO17_Pressure':'SJ_pressure',
    'KCABAKER271_Temperature': 'BAKE_temp',
    'KCABAKER271_Humidity':'BAKE_humidity',
    'KCABAKER271_Speed':'BAKE_windspeed',
    'KCABAKER271_Pressure':'BAKE_pressure',
    'KCAELSEG23_Temperature': 'EL_temp',
    'KCAELSEG23_Dew_Point':'EL_dew',
    'KCAELSEG23_Humidity':'EL_humidity',
    'KCAELSEG23_Speed':'EL_windspeed',
    'KCAELSEG23_Gust':'EL_windgust',
    'KCAELSEG23_Pressure':'EL_pressure',
    'KCARIVER117_Temperature': 'RIV_temp',
    'KCARIVER117_Dew_Point':'RIV_dew',
    'KCARIVER117_Humidity':'RIV_humidity',
    'KCARIVER117_Speed':'RIV_windspeed',
    'KCARIVER117_Gust':'RIV_windgust',
    'KCARIVER117_Pressure':'RIV_pressure'
}

# Columns for swap missing NaN data between SF and SJ
sf_columns = [
    'KCASANFR698_Temperature', 'KCASANFR698_Dew_Point', 'KCASANFR698_Humidity',
    'KCASANFR698_Speed', 'KCASANFR698_Gust', 'KCASANFR698_Pressure'
]

sj_columns = [
    'KCASANJO17_Temperature', 'KCASANJO17_Dew_Point', 'KCASANJO17_Humidity',
    'KCASANJO17_Speed', 'KCASANJO17_Gust', 'KCASANJO17_Pressure'
]


# Specify the date ranges
train_start_date = pd.to_datetime('2021-01-02')
train_end_date = pd.to_datetime('2023-10-03')
predict_date = pd.to_datetime('2023-10-04')

# Assuming you want date objects
train_start_date_date = train_start_date.date()
train_end_date_date = train_end_date.date()
predict_date_date = predict_date.date()

#library, change to file path where .csv files located
path = 'C:/Users/groutgauss/Machine_Learning_Projects/CAISO Price Forecast/Machine Learning/'

merged_df = pd.read_csv(path + 'data.csv')

merged_df = swap_missing_data(merged_df, sf_columns, sj_columns) #Swap SF and SJ weather data for NaN values
merged_df = interpolate_missing(merged_df)
data = rename_columns(merged_df, column_mapping) #renames the column headers
date_and_hour(data)
data = add_holiday_variable(data, 'datetime', train_start_date, train_end_date) # Add holiday variable

#filter train/test data for start/end dates
df = data[(data['datetime'] >= train_start_date) & (data['datetime'] <= train_end_date + pd.Timedelta(days=1))]

# Split data into features and target
X = df.drop(['datetime','TH_SP15_GEN-APND'], axis=1).values
y = df['TH_SP15_GEN-APND'].values

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Initialize dictionaries to store forecasts
hourly_forecasts = {model_name: [] for model_name in models}

##Model fit##
# Loop through models and hours for training
for model_name in models:
    for hour in range(1, 25):
        
        X_train_hour = X_train[X_train['he'] == hour].drop(['he', 'date'], axis=1)
        y_train_hour = y_train[X_train['he'] == hour]
        
        model = train_model(model_name, X_train_hour, y_train_hour)
        
        # Assuming you have 2023-07-31 data for exogenous features
        X_forecast = data[data['date'] == pd.to_datetime(predict_date_date).date()]
        
        # Assuming you have 2023-07-31 data for exogenous features
        X_forecast = data[data['date'] == pd.to_datetime(predict_date_date).date()]
        X_forecast = X_forecast[X_forecast['he'] == hour].drop(['datetime', 'TH_SP15_GEN-APND', 'date', 'he'], axis=1)
        y_forecast = model.predict(X_forecast)
              
        hourly_forecasts[model_name].append(y_forecast)
        

##Compare Performance##

# Subset df for the predict_date
actuals = data[data['date'] == pd.to_datetime(predict_date_date)]
actuals = actuals[['date', 'he', 'TH_SP15_GEN-APND']]
actuals = actuals.reset_index(drop=True)
actual_price = actuals['TH_SP15_GEN-APND'] # Extract actual load data

# Prediction vs Actuals
plt.figure(figsize=(15, 6))  # Set figure size to 15 x 6
plt.plot(range(24), actual_price, color='red', marker='o', label='Actual SP15 Price') # Plot the actual load in red

# Plot model predictions
for model_name, forecasts in hourly_forecasts.items():
    plt.plot(range(24), forecasts[:24], label=f'{model_name} Forecast', linestyle='dotted')

# Add plot elements
plt.title("Model Predictions vs. Actual Load")
plt.xlabel("Hour")
plt.ylabel("Load")
plt.legend(loc="upper left")

plt.show()