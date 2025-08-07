import pandas as pd
from sqlalchemy import create_engine
import psycopg2 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

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


table_name_raw = 'superstore_raw'
query = f"SELECT order_date, sales FROM {table_name_raw} WHERE sales IS NOT NULL AND order_date IS NOT NULL ORDER BY order_date;"

try:
    
    df_from_db = pd.read_sql(query, engine, parse_dates=['order_date'])
    df_from_db.set_index('order_date', inplace=True)
    print(f"\nSUCCESS: Retrieved {len(df_from_db)} records from '{table_name_raw}' in PostgreSQL.")
    print("--- Data from DB Head (first 5 rows) ---")
    print(df_from_db.head())
    print("--- Data from DB Info ---")
    print(df_from_db.info())
except Exception as e:
    print(f"ERROR: Failed to retrieve data from PostgreSQL table '{table_name_raw}'.")
    print(f"Details: {e}")
    exit()

# --- Aggregating to Monthly Sales ---
monthly_sales_df = df_from_db['sales'].resample('MS').sum().to_frame()

monthly_sales_df.columns = ['total_monthly_sales']

monthly_sales_df.fillna(0, inplace=True)

print(f"\nSUCCESS: Data aggregated to monthly sales. Total months: {len(monthly_sales_df)}")
print(f"Aggregated data range: {monthly_sales_df.index.min().strftime('%Y-%m-%d')} to {monthly_sales_df.index.max().strftime('%Y-%m-%d')}")
print("--- Monthly Aggregated Sales Data (First 5 rows) ---")
print(monthly_sales_df.head())
print("--- Monthly Aggregated Sales Data (Last 5 rows) ---")
print(monthly_sales_df.tail())
print("--- Monthly Aggregated Sales Data Info ---")
print(monthly_sales_df.info())

# --- Creating Time-Based Features ---

monthly_sales_df['year'] = monthly_sales_df.index.year
monthly_sales_df['month'] = monthly_sales_df.index.month       
monthly_sales_df['quarter'] = monthly_sales_df.index.quarter   
monthly_sales_df['month_name'] = monthly_sales_df.index.strftime('%B') 
monthly_sales_df['day_of_month'] = monthly_sales_df.index.day  

print("\nSUCCESS: Added time-based features.")
print("--- Monthly Sales with Time-Based Features (First 5 rows) ---")
print(monthly_sales_df.head())

# --- Creating Lag Features ---
monthly_sales_df['lag_1_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(1)
monthly_sales_df['lag_3_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(3)
monthly_sales_df['lag_6_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(6)
monthly_sales_df['lag_12_month_sales'] = monthly_sales_df['total_monthly_sales'].shift(12) 

print("\nSUCCESS: Added lag features.")
print("--- Monthly Sales with Lag Features (First 15 rows to see lags fill in) ---")
print(monthly_sales_df.head(15)) 

# --- Creating Rolling Statistics (Moving Averages, Standard Deviations) ---
monthly_sales_df['rolling_mean_3_month'] = monthly_sales_df['total_monthly_sales'].rolling(window=3).mean()
monthly_sales_df['rolling_std_3_month'] = monthly_sales_df['total_monthly_sales'].rolling(window=3).std()
monthly_sales_df['rolling_mean_6_month'] = monthly_sales_df['total_monthly_sales'].rolling(window=6).mean()
monthly_sales_df['rolling_std_6_month'] = monthly_sales_df['total_monthly_sales'].rolling(window=6).std()

print("\nSUCCESS: Added rolling statistics features.")
print("--- Monthly Sales with Rolling Features (First 10 rows) ---")
print(monthly_sales_df.head(10))

# --- Handling NaN values introduced by Feature Engineering ---
initial_rows_count = len(monthly_sales_df)
monthly_sales_df.dropna(inplace=True) 
print(f"\nINFO: Original rows before dropping NaNs from feature engineering: {initial_rows_count}")
print(f"INFO: Rows after dropping NaNs (due to lag/rolling features): {len(monthly_sales_df)}")
print(f"INFO: Number of rows dropped: {initial_rows_count - len(monthly_sales_df)}")

print("\n--- Final DataFrame with Engineered Features (First 5 rows after dropping NaNs) ---")
print(monthly_sales_df.head())
print("--- Final DataFrame Info after Feature Engineering ---")
print(monthly_sales_df.info())

# --- Defining Target (y) and Features (X) ---
target_column = 'total_monthly_sales'

features_columns = [col for col in monthly_sales_df.columns if col not in [target_column, 'month_name', 'day_of_month']]

X = monthly_sales_df[features_columns]
y = monthly_sales_df[target_column]

print(f"\nSUCCESS: Defined features (X) and target (y).")
print(f"Features (X) include: {features_columns}")
print(f"Target (y) is: '{target_column}'")

# --- Chronological Split into Training and Testing Sets ---
forecast_horizon_months = 12 

# Calculating the size of the training set
train_size = len(monthly_sales_df) - forecast_horizon_months

# Spliting the DataFrame by index location (iloc)
train_df = monthly_sales_df.iloc[:train_size]
test_df = monthly_sales_df.iloc[train_size:]

# Spliting features (X) and target (y) for both train and test sets
X_train = train_df[features_columns]
y_train = train_df[target_column]

X_test = test_df[features_columns]
y_test = test_df[target_column]

print(f"\nSUCCESS: Data split chronologically.")
print(f"Total months in cleaned data: {len(monthly_sales_df)}")
print(f"Training set months: {len(train_df)} (from {train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
print(f"Testing set months: {len(test_df)} (from {test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")

print("\nX_train (first 3 rows):")
print(X_train.head(3))
print("\ny_train (first 3 rows):")
print(y_train.head(3))
print("\nX_test (first 3 rows):")
print(X_test.head(3))
print("\ny_test (first 3 rows):")
print(y_test.head(3))

print("\nData Preprocessing & Feature Engineering complete!")