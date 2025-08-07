import pandas as pd
from sqlalchemy import create_engine
import psycopg2 

# --- Configuration for PostgreSQL ---
DB_HOST = 'localhost'         
DB_NAME = 'sales_forecast_db' 
DB_USER = 'dashboard_user'    
DB_PASSWORD = 'salesforecasting' 

# Creating a SQLAlchemy engine for connecting to PostgreSQL
try:
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
    print("PostgreSQL engine created successfully. Ready for database interaction.")
except Exception as e:
    print(f"ERROR: Could not create PostgreSQL engine. Check your DB_HOST, DB_NAME, DB_USER, DB_PASSWORD.")
    print(f"Details: {e}")
    print("Please ensure PostgreSQL is running and connection details are correct (password, user, database name).")
    exit()


file_path = 'Sample - Superstore.csv'

# --- Loading the dataset into a Pandas DataFrame ---
try:
    df_raw = pd.read_csv(file_path, encoding='latin1')
    print(f"\nSUCCESS: Loaded data from {file_path}. Initial shape: {df_raw.shape} (rows, columns).")
except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found.")
    exit()
except Exception as e:
    print(f"ERROR: An unexpected error occurred while loading the CSV: {e}")
    exit()

# --- Initial Data Inspection ---
print("\n--- Initial DataFrame Head (first 5 rows) ---")
print(df_raw.head())
print("\n--- Initial DataFrame Info (data types and non-null counts) ---")
print(df_raw.info())
print("\n--- All Column Names ---")
print(df_raw.columns.tolist()) 

expected_date_col = 'Order Date'
expected_sales_col = 'Sales'
if expected_date_col not in df_raw.columns:
    print(f"ERROR: Expected column '{expected_date_col}' not found.")
    exit()
if expected_sales_col not in df_raw.columns:
    print(f"ERROR: Expected column '{expected_sales_col}' not found.")
    exit()
print(f"\nCONFIRMED: '{expected_date_col}' and '{expected_sales_col}' columns are present.")

# --- Initial Data Cleaning (Transforming for PostgreSQL ingestion) ---
df_raw.columns = df_raw.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('.', '', regex=False)
print("\nSUCCESS: Column names standardized for SQL.")
print("New columns:", df_raw.columns.tolist())

df_raw['order_date'] = pd.to_datetime(df_raw['order_date'], errors='coerce')
initial_null_dates_count = df_raw['order_date'].isnull().sum()
if initial_null_dates_count > 0:
    print(f"WARNING: Found {initial_null_dates_count} null 'order_date' values after conversion. These rows will be dropped.")
    df_raw.dropna(subset=['order_date'], inplace=True) 
    print(f"INFO: DataFrame shape after dropping null dates: {df_raw.shape}")
else:
    print("INFO: No null 'order_date' values found after conversion.")

df_raw['sales'] = pd.to_numeric(df_raw['sales'], errors='coerce')
initial_null_sales_count = df_raw['sales'].isnull().sum()
if initial_null_sales_count > 0:
    print(f"WARNING: Found {initial_null_sales_count} null 'sales' values after conversion. These will be filled with 0.")
    df_raw['sales'].fillna(0, inplace=True) 
else:
    print("INFO: No null 'sales' values found after conversion.")

print("\nSUCCESS: 'order_date' and 'sales' columns processed for data types and nulls.")
print("--- DataFrame Info after initial cleaning ---")
print(df_raw.info())

# --- Ingesting Cleaned Raw Data into PostgreSQL ---
table_name_raw = 'superstore_raw'
try:
    df_raw.to_sql(table_name_raw, engine, if_exists='replace', index=False)
    print(f"\nSUCCESS: Data successfully ingested into '{table_name_raw}' table in PostgreSQL.")
    print(f"Total rows ingested: {len(df_raw)}")
except Exception as e:
    print(f"ERROR: Failed to ingest data into PostgreSQL table '{table_name_raw}'.")
    print(f"Details: {e}")
    print("Please check your database connection, user permissions, and ensure PostgreSQL is running.")
    exit()

# ---  Verification ---

import psycopg2
try:
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name_raw};")
    row_count = cur.fetchone()[0]
    print(f"\nVERIFICATION: Confirmed {row_count} rows in '{table_name_raw}' table in PostgreSQL.")
    cur.close()
    conn.close()
except Exception as e:
    print(f"ERROR: Could not verify data in PostgreSQL: {e}")

print("\n Superstore data is now in PostgreSQL!")