# Sales-Forecasting-Dashboard-with-Time-Series-Analysis
This project provides a comprehensive solution for sales forecasting, leveraging Python for data processing and modeling, PostgreSQL for data storage, and Power BI for interactive data visualization. The goal is to provide a robust, end-to-end tool that enables a data-driven approach to sales strategy and performance evaluation. The dataset is sourced from Kaggle.

## Project Architecture
The project follows a standard data pipeline, beginning with raw data ingestion and concluding with an interactive dashboard. The key components are:

### Data Cleaning and Ingestion (data_cleaning_and_ingestion.py): 
This script reads a raw CSV file, standardizes column names, and cleans the data by handling null values and converting data types. The cleaned data is then ingested into a PostgreSQL database, ensuring a single, reliable source of truth. The script specifically processes order_date and sales columns, converting order_date to a datetime object and filling any null sales values with 0.

### Data Preprocessing and Feature Engineering (data_preprocessing_and_features.py):
This script retrieves the cleaned data from PostgreSQL, aggregates it to a monthly level, and engineers new features critical for time series modeling. These features include year, month, quarter, and rolling_mean_3_month. The data is then split chronologically into training and testing sets, preparing it for the forecasting model.

### Model Development (model_development.py): 
The core of the forecasting is done here using the Prophet library. The script trains a model on the prepared data, including custom regressors like year, month, and quarter. It then evaluates the model's performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) and saves the trained model as a .joblib file for future use.

### Data Integration for Power BI (data_integration_for_powerbi.py): 
This final Python script loads the trained Prophet model, generates a full forecast including future predictions, and merges it with the historical sales data. It creates a Type column to distinguish between 'Actual' and 'Forecast' data points, making it ready for visualization. The final, combined dataset is then ingested into a dedicated table in PostgreSQL, which serves as the data source for the Power BI dashboard.

## Power BI Dashboard: Key Insights
The Power BI dashboard consists of four pages, each providing unique insights for different business needs.

### 1. Executive Summary
This page provides a quick, high-level overview of the sales landscape. It shows a Total Actual Sales figure of **2.30M** and a Total Forecasted Sales of **2.59M**. A key insight is the projected Year-over-Year Growth Forecast of **21.20%**, indicating a strong growth trajectory. The "Monthly Sales Performance" chart highlights a clear seasonal pattern with sales consistently peaking towards the end of each year.

### 2. Forecast Deep-Dive
This page is designed to help users understand the components of the forecast. The "Forecast Components: Trend & Seasonality" chart is crucial, as it separates the forecast into a long-term upward trend and a cyclical yearly seasonal pattern. The seasonality component fluctuates by approximately 1.0 unit, confirming that sales are predictably higher during certain months. The "Forecast with 80% Confidence Interval" provides a clear range of possible outcomes, which is vital for risk assessment and setting realistic expectations. The dashboard also provides immediate, actionable numbers like the Average Monthly Forecast of **53.96K** and the Next Month's Forecast of **59.83K**.

### 3. Seasonal Trends & Performance
This page moves beyond simple time-series analysis to provide a direct comparison of performance. The "Monthly Sales by Year" bar chart allows for easy year-over-year analysis, showing a significant increase in forecast for November from **76K** in **2015** to **126K** in **2018**. The "Sales by Month-of-Year" donut chart provides a proportional breakdown, revealing that December and March account for **16.24%** and **9.28%** of total sales, respectively. This insight is essential for resource allocation and targeted marketing.

### 4. Actual vs. Forecast Performance
This is the most critical page for evaluating the model's accuracy and business performance. The Sales to Forecast Variance of **-293.03K** is a key KPI indicating that actual sales were **$293,030** less than the total forecast. This underperformance is visually confirmed in the "Monthly Actual vs. Forecast" combo chart, which shows periods where the forecast line was consistently above the actual sales bars. The "Monthly Data Summary" table provides the granular data needed to identify the specific months where the largest variances occurred, enabling a targeted root-cause analysis.

## Conclusion
This project successfully integrates Python, PostgreSQL, and Power BI to create a powerful and comprehensive sales forecasting and performance analysis tool. The Python scripts handle the entire data and modeling pipeline, from initial cleaning and feature engineering to generating a robust Prophet-based forecast. PostgreSQL serves as the central data repository, ensuring data integrity and accessibility. Finally, the Power BI dashboard leverages this structured data to deliver a rich, interactive experience with multiple pages, allowing stakeholders to not only see the forecast but also to understand its components, evaluate its accuracy, and gain actionable insights into historical performance.
