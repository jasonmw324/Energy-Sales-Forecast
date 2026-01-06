# Energy Sales Forecasting Project

## Project Background
This project analyzes historical energy sales data (in megawatt-hours) from the **U.S. Energy Information Administration (EIA)** to forecast future energy demand. Accurate energy demand forecasting supports better planning, grid management, and resource allocation. By modeling trends and seasonality in historical consumption, this project demonstrates practical applications of time series analysis for energy markets.

## Objective
The primary objective of this project is to analyze historical energy sales and forecast future demand using **SARIMA (Seasonal ARIMA)** modeling. Specifically, the project focuses on:  
- Identifying trends and seasonal patterns in historical energy sales  
- Building a forecasting model with reliable predictive performance


> The Python code and analysis can be found [here](Energy_Sales_Forecast.ipynb)

## Dataset Overview
The dataset contains historical energy sales (in MWh) from the EIA, with the following key variables:  

- **date** – timestamp of the observation  
- **sales** – total energy sold (megawatt-hours)  


## Methods

The analysis leveraged **SARIMA modeling** to capture both trend and seasonal patterns:

- **Exploratory Data Analysis (EDA):**  
  Visualized historical energy sales to detect seasonality, trend components, and anomalies. Decomposition helped determine SARIMA parameters.  

- **Alternative Approaches Considered:**  
  Simple ARIMA models were tested but underperformed due to strong seasonal effects. Holt-Winters exponential smoothing was also explored but did not capture finer seasonal cycles as well as SARIMA.

- **Final Model Selection:**  
  Two SARIMA models were built:  

  **Model 1 – Conservative:**  
  - Prioritizes steady trends in the data  
  - Produces cautious forecasts, minimizing overestimation of future growth  

  **Model 2 – Growth:**  
  - Emphasizes the upward trend observed in the last 4 years  
  - Produces higher forecasts, capturing potential growth opportunities  

- **Validation:** Residual analysis and forecast accuracy metrics (MAE, RMSE) were used to ensure reliability.  

![Forecast vs Actual](forecast_plot.png)  

## Results

- Both models capture trend and seasonal patterns in energy sales  
- **Conservative model** provides steady forecasts for cautious planning  
- **Growth model** emphasizes recent upward trends to plan for higher potential demand  
- Residual analysis confirmed minimal autocorrelation, indicating well-fitted models  

## Streamlit Dashboard
A **Streamlit dashboard** was developed to interactively explore historical energy sales and forecasts:  
- Visualize historical trends and seasonal patterns  
- Compare forecasts from both models with actuals  
- Adjust forecast horizon to simulate different planning scenarios  

## Next Steps / Future Enhancements
- Incorporate additional features such as **weather, holidays, and economic indicators** to improve forecasts  
- Explore **multivariate time series models** for region-specific energy predictions  
- Deploy the Streamlit dashboard online for interactive, real-time energy sales forecasting  

