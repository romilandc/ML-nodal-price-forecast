import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar


def date_and_hour(df):
    # Convert 'datetime' column to datetime type if not already
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract 'date' and 'hour' columns
    df['date'] = df['datetime'].dt.date
    df['he'] = (df['datetime'].dt.hour + 1) % 25  # Add 1 to hour, ensuring it stays within 0-23 range

    return df

def add_holiday_variable(data, date_column, start_date, end_date):

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    data['is_holiday'] = (data[date_column].dt.normalize().isin(holidays)).astype(int)
    return data

def create_lagged_variables(data, lagged_column, lag_range):
    
    for lag in range(1, lag_range + 1):
        data[f'{lagged_column}_lag_{lag}'] = data[lagged_column].shift(lag)
    return data

def swap_missing_data(merged_df, sf_columns, sj_columns):
    # Replace NaN values in San Francisco columns with values from San Jose columns
    for col_sf, col_sj in zip(sf_columns, sj_columns):
        merged_df[col_sf].fillna(merged_df[col_sj], inplace=True)
        merged_df[col_sj].fillna(merged_df[col_sf], inplace=True)

    return merged_df

def rename_columns(df, column_mapping):
    # Use the rename method to rename the columns
    df = df.rename(columns=column_mapping)
    return df

def interpolate_missing(df):

    result_df = df.copy()

    # Ensure 'Time' column is in datetime format
    result_df['Datetime'] = pd.to_datetime(result_df['Datetime'])

    for column in df.columns:
        if column not in ['Datetime'] and pd.api.types.is_numeric_dtype(result_df[column]):
            try:
                # Convert the column to numeric (if not already)
                result_df[column] = pd.to_numeric(result_df[column], errors='coerce')

                # Extract hour component and interpolate within each hour
                result_df[column] = result_df.groupby(result_df['Datetime'].dt.hour)[column].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both'))

            except ValueError:
                print(f"Skipping interpolation for non-numeric column: {column}")

    return result_df

def split_data(df, test_size=0.2, random_state=35):

    X = df.drop(['TH_SP15_GEN-APND', 'datetime'], axis=1)
    y = df['TH_SP15_GEN-APND']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100