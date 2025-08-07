import pandas as pd
from sqlalchemy import create_engine
import joblib
from prophet import Prophet
import numpy as np

# --- Configuration for PostgreSQL (Same as previous phases) ---
DB_HOST = 'localhost'
DB_NAME = 'sales_forecast_db'
DB_USER = 'dashboard_user'
DB_PASSWORD = 'salesforecasting' 

# Creating a SQLAlchemy engine for connecting to PostgreSQL
try:
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
    print("PostgreSQL engine created successfully. Ready for database interaction.")
except Exception as e:
    print(f"ERROR: Could not create PostgreSQL engine. Check configuration.")
    print(f"Details: {e}")
    exit()

# --- Loading the Trained Prophet Model ---
model_filename = 'sales_forecast_prophet_model.joblib'
try:
    # Loading the model 
    model = joblib.load(model_filename)
    print(f"\nSUCCESS: Prophet model loaded from '{model_filename}'.")
except FileNotFoundError:
    print(f"ERROR: Model file '{model_filename}' not found.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load the model. Details: {e}")
    exit()

# --- Generating Full Forecast and Preparing Data for Power BI ---

print("\n--- Re-fetching and processing historical data ---")
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

    
    df_prophet_input = monthly_sales_df.reset_index().rename(columns={'order_date': 'ds', 'total_monthly_sales': 'y'})


    df_prophet_input.dropna(inplace=True)

    print(f"SUCCESS: Historical data prepared. Shape: {df_prophet_input.shape}")
except Exception as e:
    print(f"ERROR: Failed during historical data preparation. Details: {e}")
    exit()

# Defining the forecast horizon in months
forecast_horizon_months = 24

# Creating a DataFrame with all dates for full forecast
future_dates = model.make_future_dataframe(
    periods=forecast_horizon_months,
    freq='MS',
    include_history=True
)

# Adding regressors to the future_dates DataFrame for both historical and future periods
future_dates['year'] = future_dates['ds'].dt.year
future_dates['month'] = future_dates['ds'].dt.month
future_dates['quarter'] = future_dates['ds'].dt.quarter


print("\n--- Columns in future_dates before model.predict---")
print(future_dates.columns)



# Making the full forecast
print("\n--- Generating full forecast ---")
full_forecast = model.predict(future_dates)
print("SUCCESS: Full forecast generated.")

print("\n--- Columns in full_forecast after model.predict ---")
print(full_forecast.columns)



# --- Preparing the Combined Data for Power BI ---


forecast_pbi_df = full_forecast[[
    'ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly' 
    
]].copy()



print(f"forecast_pbi_df shape before rename: {forecast_pbi_df.shape}")
print(f"forecast_pbi_df head before rename:\n{forecast_pbi_df.head()}")
print(f"forecast_pbi_df tail before rename:\n{forecast_pbi_df.tail()}")



# Renaming columns 
forecast_pbi_df.rename(columns={
    'ds': 'Date',
    'yhat': 'Forecasted_Sales',
    'yhat_lower': 'Forecast_Lower_Bound',
    'yhat_upper': 'Forecast_Upper_Bound',
    'trend': 'Forecast_Trend',
    'yearly': 'Forecast_Yearly_Seasonality'
}, inplace=True)



actual_sales_df = df_prophet_input[['ds', 'y']].rename(columns={'ds': 'Date', 'y': 'Actual_Sales'})


print(f"actual_sales_df shape: {actual_sales_df.shape}")
print(f"actual_sales_df head:\n{actual_sales_df.head()}")
print(f"actual_sales_df tail:\n{actual_sales_df.tail()}")


combined_data_for_pbi = pd.merge(forecast_pbi_df, actual_sales_df, on='Date', how='outer')


print(f"combined_data_for_pbi shape after merge: {combined_data_for_pbi.shape}")
print(f"combined_data_for_pbi head after merge:\n{combined_data_for_pbi.head()}")
print(f"combined_data_for_pbi tail after merge:\n{combined_data_for_pbi.tail()}")



last_actual_date = df_prophet_input['ds'].max()

print("\n--- Combined Data for Power BI (Head) ---")
print(combined_data_for_pbi.head())
print("\n--- Combined Data for Power BI (Tail - to see forecasts) ---")
print(combined_data_for_pbi.tail())
print("\n--- Combined Data for Power BI (Info) ---")
print(combined_data_for_pbi.info())


combined_data_for_pbi['Type'] = combined_data_for_pbi['Actual_Sales'].apply(
    lambda x: 'Forecast' if pd.isna(x) else 'Actual'
)


print("\n--- Combined Data for Power BI (Head) ---")
print(combined_data_for_pbi.head())

# --- Ingesting Combined Data into PostgreSQL for Power BI ---
table_name_pbi = 'sales_forecast_for_powerbi' 

try:
    combined_data_for_pbi.to_sql(table_name_pbi, engine, if_exists='replace', index=False)
    print(f"\nSUCCESS: Combined data successfully ingested into '{table_name_pbi}' table in PostgreSQL.")
    print(f"Total rows ingested: {len(combined_data_for_pbi)}")
except Exception as e:
    print(f"ERROR: Failed to ingest combined data into PostgreSQL table '{table_name_pbi}'.")
    print(f"Details: {e}")
    exit()

print("\n Data Integration for Power BI complete!")