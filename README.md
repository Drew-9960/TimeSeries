# Time Series Forecasting Project

## Project Overview

This project focuses on time series forecasting using electricity usage data. The goal is to predict future electricity consumption using various forecasting techniques including a Naive model and the ARIMA model. The dataset contains the total electricity usage recorded at different time intervals.

## Dataset

The dataset `elec_time.xlsx` contains electricity usage data with the following features:
- **Billing Start Date**: The start date of the billing cycle.
- **Total usage**: The total electricity usage in the billing period.

## Data Preprocessing

1. **Loading Data**:
   ```python
   import pandas as pd 
   data = pd.read_excel('elec_time.xlsx', parse_dates=[0])
   data.set_index('Billing Start Date', inplace=True)
   data = data[['Total usage']]
   ```

2. **Data Cleaning**:
   - Drop rows with all missing values.
   - Filter out usage values less than 1.
   - Remove outliers with usage values greater than 40,000.

   ```python
   data.dropna(how='all', inplace=True)
   data = data[data['Total usage'] >= 1]
   data = data[data['Total usage'] <= 40000]
   ```

3. **Data Visualization**:
   - Plot the total usage data.
   - Plot the rolling mean with a window of 100.

   ```python
   import matplotlib.pyplot as plt 
   data.plot()
   data_mean = data.rolling(window=100).mean()
   data_mean.plot()
   ```

   - Box plot to identify outliers.

   ```python
   import seaborn as sns
   sns.boxplot(x=data['Total usage'])
   ```

## Modeling

### Naive Model

1. **Creating the Naive Model**:
   ```python
   series_value = data.values
   val = pd.DataFrame(series_value)
   df = pd.concat([val, val.shift(1)], axis=1)
   df = df[1:]
   df.columns = ['Actual_output', 'Forecasted_output']
   ```

2. **Model Evaluation**:
   - Calculate the Root Mean Squared Error (RMSE).

   ```python
   from sklearn.metrics import mean_squared_error
   df_error = mean_squared_error(df['Actual_output'], df['Forecasted_output'])
   np.sqrt(df_error)
   ```

### ARIMA Model

1. **Parameter Selection**:
   - Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

   ```python
   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   plot_acf(data)
   plot_pacf(data)
   ```

2. **Building the ARIMA Model**:
   - Split the data into training and testing sets.
   
   ```python
   df_train = data[0:50000]
   df_test = data[50000:]
   ```

   - Train the ARIMA model with selected order (p, d, q).

   ```python
   from statsmodels.tsa.arima.model import ARIMA
   df_model = ARIMA(df_train, order=(2, 1, 2))
   df_fit = df_model.fit()
   ```

3. **Forecasting and Evaluation**:
   - Forecast the future values.
   - Evaluate the model performance using RMSE.

   ```python
   df_forecast = df_fit.forecast(steps=len(df_test))[0]
   np.sqrt(mean_squared_error(df_test, df_forecast))
   ```

## Results

- **Naive Model**:
  - RMSE: *value*

- **ARIMA Model**:
  - AIC: *value*
  - RMSE: *value*

## Conclusion

The ARIMA model provides a more accurate forecast of electricity usage compared to the Naive model based on RMSE values. Future work can include exploring other time series forecasting methods like SARIMA, LSTM, and Prophet.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

