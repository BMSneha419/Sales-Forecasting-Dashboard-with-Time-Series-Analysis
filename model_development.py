import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet 
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import numpy as np
import matplotlib.pyplot as plt
import joblib 

# --- Configuration for PostgreSQL  ---
DB_HOST = 'localhost'
DB_NAME = 'sales_forecast_db'
DB_USER = 'dashboard_user'
DB_PASSWORD = 'salesforecasting' 

# Creating a SQLAlchemy engine for connecting to PostgreSQL
try:
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
    print("PostgreSQL engine created successfully.")
except Exception as e:
    print(f"ERROR: Could not create PostgreSQL engine. Check configuration.")
    print(f"Details: {e}")
    exit()

# --- Loading & Preparing Data for Prophet ---

print("\n--- Re-fetching and processing data for Prophet ---")
table_name_raw = 'superstore_raw'
query = f"SELECT order_date, sales FROM {table_name_raw} WHERE sales IS NOT NULL AND order_date IS NOT NULL ORDER BY order_date;"

try:
    df_from_db = pd.read_sql(query, engine, parse_dates=['order_date'])
    df_from_db.set_index('order_date', inplace=True)
    monthly_sales_df = df_from_db['sales'].resample('MS').sum().to_frame()
    monthly_sales_df.columns = ['total_monthly_sales']
    monthly_sales_df.fillna(0, inplace=True)

    monthly_sales_df['year'] = monthly_sales_df.index.year
    monthly_sales_df['month'] = monthly_sales_df.index.month
    monthly_sales_df['quarter'] = monthly_sales_df.index.quarter
    monthly_sales_df['lag_1_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(1)
    monthly_sales_df['lag_12_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(12)
    monthly_sales_df['rolling_mean_3_month'] = monthly_sales_df['total_monthly_sales'].rolling(window=3).mean()

    # Droping NaNs created by lag/rolling features
    initial_rows_count = len(monthly_sales_df)
    monthly_sales_df.dropna(inplace=True)
    if len(monthly_sales_df) < initial_rows_count:
        print(f"INFO: Dropped {initial_rows_count - len(monthly_sales_df)} rows with NaNs from feature engineering.")
    else:
        print("INFO: No NaNs dropped after feature engineering.")


    print(f"SUCCESS: Monthly sales data with features prepared. Final shape: {monthly_sales_df.shape}")
except Exception as e:
    print(f"ERROR: Failed during data preparation for Prophet. Details: {e}")
    exit()

# Converting the index to a column for 'ds'
df_prophet = monthly_sales_df.reset_index()
df_prophet = df_prophet.rename(columns={'order_date': 'ds', 'total_monthly_sales': 'y'})

print("\n--- Data prepared for Prophet (first 5 rows) ---")
print(df_prophet.head())
print("--- Data prepared for Prophet Info ---")
print(df_prophet.info())

# --- Chronological Split for Prophet ---
forecast_horizon_months = 12
train_size = len(df_prophet) - forecast_horizon_months

train_prophet_df = df_prophet.iloc[:train_size]
test_prophet_df = df_prophet.iloc[train_size:] 

print(f"\nData split into training ({len(train_prophet_df)} months) and testing ({len(test_prophet_df)} months).")
print(f"Training Period: {train_prophet_df['ds'].min().strftime('%Y-%m-%d')} to {train_prophet_df['ds'].max().strftime('%Y-%m-%d')}")
print(f"Testing Period: {test_prophet_df['ds'].min().strftime('%Y-%m-%d')} to {test_prophet_df['ds'].max().strftime('%Y-%m-%d')}")


# --- Training Prophet Model ---
print("\n--- Training Prophet Model ---")
model = Prophet(
    growth='linear',
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.95,
)
model.add_seasonality('yearly', period=365.25, fourier_order=10)

model.add_regressor('year')
model.add_regressor('month')
model.add_regressor('quarter')

print("\n--- Prophet Model Seasonality Components ---")
print(model.seasonalities) 
# Fitting the model to your training data
model.fit(train_prophet_df)
print("SUCCESS: Prophet model trained with regressors.")

# ---  Making Future Predictions ---
print("\n--- Making Future Predictions ---")
future = model.make_future_dataframe(periods=forecast_horizon_months, freq='MS')
future['year'] = future['ds'].dt.year
future['month'] = future['ds'].dt.month
future['quarter'] = future['ds'].dt.quarter

forecast = model.predict(future)


forecast_test_period = forecast[forecast['ds'].isin(test_prophet_df['ds'])]

print("SUCCESS: Predictions made for the test period.")
print("--- Forecasted Test Period Head ---")
print(forecast_test_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# ---  Evaluating Model Performance ---
print("\n--- Evaluating Model Performance ---")
evaluation_df = pd.merge(test_prophet_df, forecast_test_period[['ds', 'yhat']], on='ds', how='left')

# Calculating common evaluation metrics
mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
mse = mean_squared_error(evaluation_df['y'], evaluation_df['yhat'])
rmse = np.sqrt(mse) 

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# --- Visualizing Results ---
print("\n--- Visualizing Forecast Results ---")
fig1 = model.plot(forecast)
plt.title('Prophet Forecast with Components')
plt.xlabel('Date')
plt.ylabel('Sales')

plt.plot(train_prophet_df['ds'], train_prophet_df['y'], 'o', color='blue', label='Training Actuals', markersize=4)

plt.plot(test_prophet_df['ds'], test_prophet_df['y'], 'o', color='red', label='Test Actuals', markersize=4)

plt.legend()
plt.show()

# Plotting Prophet components
fig2 = model.plot_components(forecast)
plt.show()

# --- Saving the Trained Model ---
model_filename = 'sales_forecast_prophet_model.joblib'
try:
    joblib.dump(model, model_filename)
    print(f"\nSUCCESS: Trained Prophet model saved as '{model_filename}'.")
except Exception as e:
    print(f"ERROR: Could not save the model. Details: {e}")

print("\nTime Series Model Development complete!")