# Energy Sales Forecasting Project

## Executive Summary

Energy sales remained relatively stable from 2010-2019, fluctuating around a modest upward trend. However, starting in early 2020, sales began increasing significantly, creating uncertainty about whether this represents a permanent shift or temporary spike.

[**Visual placeholder**: Line chart showing historical energy sales 2010-2024, with annotation highlighting the 2020 inflection point]

To address this uncertainty, two forecasting scenarios were developed:

- **Growth Model**: Assumes the recent upward trend continues, helping plan for higher demand scenarios (95.4% accuracy)
- **Conservative Model**: Assumes reversion to the historical stable pattern, supporting cautious resource planning (97.2% accuracy)

Both models capture seasonal patterns effectively, with residual analysis confirming well-fitted forecasts. This dual-scenario approach enables decision-makers to evaluate risk and plan for multiple possible futures.

*Accuracy calculated as 100 - MAPE, meaning forecasts are within 2.5% of actual values on average.

> 📊 **View the full analysis:** [Python notebook](Energy_Sales_Forecast.ipynb)

> 📊 **View interactive dashboard:** [Streamlit App](https://energy-sales-forecast.streamlit.app/)


---

## Project Background

This project analyzes historical energy sales data (in megawatt-hours) from the **U.S. Energy Information Administration (EIA)** to forecast future energy demand. Accurate energy demand forecasting supports better planning, grid management, and resource allocation. By modeling trends and seasonality in historical consumption, this project demonstrates practical applications of time series analysis for energy markets.

## Objective

The primary objective of this project is to analyze historical energy sales and forecast future demand. Specifically, the project focuses on:  
- Identifying trends and seasonal patterns in historical energy sales  
- Building a forecasting model with reliable predictive performance

## Dataset Overview

The dataset contains historical energy sales (in MWh) from the EIA, with the following key variables:  
- **date** – timestamp of the observation  
- **sales** – total energy sold (megawatt-hours)  

---

## Key Insights

### Historical Trends
[Expand here with specific observations about the 2010-2019 pattern and the 2020+ shift - include visualizations]

### Model Performance
Both SARIMA models achieved 97.5% accuracy (100 - MAPE), with residual analysis confirming minimal autocorrelation and well-fitted forecasts.

[Add: Visual comparison of model forecasts vs actuals, residual plots if available]

---

## Methods

The analysis leveraged **SARIMA modeling** to capture both trend and seasonal patterns:

**Exploratory Data Analysis (EDA):**  
Visualized historical energy sales to detect seasonality, trend components, and anomalies. Decomposition helped determine SARIMA parameters.  

**Final Model Selection:**  
Based on the results of the exploratory data analysis (EDA) and the insights gained from applying the Auto ARIMA function, two SARIMA models were developed to capture different forecasting scenarios:

- **Model 1 – Conservative:**  
  - Prioritizes steady trends in the data  
  - Produces cautious forecasts, minimizing overestimation of future growth  

- **Model 2 – Growth:**  
  - Emphasizes the upward trend observed in the last 4 years  
  - Produces higher forecasts, capturing potential growth opportunities  

**Validation:**  
Residual analysis and forecast accuracy metrics (MAE, RMSE, MAPE) were used to ensure reliability.  

---

## Streamlit Dashboard

A **Streamlit dashboard** was developed to interactively explore historical energy sales and forecasts:  
- Visualize historical trends and seasonal patterns  
- Compare forecasts from both models with actuals  
- Adjust forecast horizon to simulate different planning scenarios  

[Add: Link to live dashboard if deployed, or screenshots/demo video]

---

## Business Applications & Recommendations

[Section to add - suggestions:]

**For Capacity Planning:**
- Use the growth model when evaluating infrastructure investments and long-term resource commitments
- Conservative model provides a lower bound for must-meet demand scenarios

**For Risk Management:**
- The divergence between models quantifies uncertainty in planning assumptions
- Monitor actual sales quarterly against both forecasts









