# -*- coding: utf-8 -*-
"""
Energy Sales Forecast Dashboard
Data Source: U.S. Energy Information Administration (EIA)
"""
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Energy Sales Forecast",
    page_icon="⚡",
    layout="wide"
)

# ----------------------------
# Load and preprocess data
# ----------------------------
@st.cache_data
def load_data():
    energy_data = pd.read_excel("DATA/sales_revenue.xlsx")
    
    energy_data = energy_data.drop(index=0).reset_index(drop=True)
    energy_data.columns = energy_data.iloc[0]
    energy_data = energy_data.iloc[1:].reset_index(drop=True)
    energy_data = energy_data.iloc[:, [0,1,2,20,21,22,23]]
    energy_data.drop(index=9690, inplace=True)
    
    energy_data = energy_data.rename(columns = {
        'Thousand Dollars' : 'revenue',
        'Megawatthours' : 'sales',
        'Count' : 'customer_count',
        'Cents/kWh' : 'price'
    })
    energy_data.columns = energy_data.columns.str.strip().str.lower()
    
    vars = ['year', 'month' , 'revenue' , 'sales' , 'customer_count' , 'price']
    for col in vars:
        energy_data[col] = pd.to_numeric(energy_data[col], errors='coerce')
    
    # Aggregate national data
    energy_data = energy_data.groupby(['year','month'], as_index=False).sum()
    if 'state' in energy_data.columns:
        energy_data.drop(columns=['state'], inplace=True)
    
    # Datetime index
    energy_data['date'] = pd.to_datetime(energy_data[['year','month']].assign(day=1))
    energy_data.set_index('date', inplace=True)
    
    return energy_data

energy_data = load_data()

# ----------------------------
# Train/Test Split and Fit Models
# ----------------------------
@st.cache_resource
def fit_models(data):
    # Split data: train up to 2022, test from 2023 onwards
    train_data = data[data['year'] <= 2022][['sales']].copy()
    test_data = data[data['year'] > 2022][['sales']].copy()
    
    # Model 1 - Conservative (train on training data only)
    scaler_1 = StandardScaler()
    sales_scaled_array_1 = scaler_1.fit_transform(train_data[['sales']])
    sales_scaled_1 = pd.Series(sales_scaled_array_1.flatten(), index=train_data.index, name='sales')
    
    model_1 = SARIMAX(
        sales_scaled_1,
        order=(1,0,0),
        seasonal_order=(2,1,0,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    # Model 2 - Growth (train on training data only)
    scaler_2 = StandardScaler()
    sales_scaled_array_2 = scaler_2.fit_transform(train_data[['sales']])
    sales_scaled_2 = pd.Series(sales_scaled_array_2.flatten(), index=train_data.index, name='sales')
    
    model_2 = SARIMAX(
        sales_scaled_2,
        order=(1,1,1),
        seasonal_order=(2,1,0,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    
    return model_1, scaler_1, model_2, scaler_2, train_data, test_data

model_1, scaler_1, model_2, scaler_2, train_data, test_data = fit_models(energy_data)

# ----------------------------
# Header
# ----------------------------
st.title("⚡ U.S. Energy Sales Forecasting Dashboard")
st.markdown("""
This dashboard forecasts U.S. electricity sales using **SARIMA** time series modeling on EIA data.
Two different modeling approaches are presented to capture varying assumptions about future trends.
""")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Forecast Settings")
n_future = st.sidebar.slider(
    "Forecast Horizon (months)", 
    min_value=1, 
    max_value=60, 
    value=24, 
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Dashboard")
st.sidebar.info("""
This tool forecasts U.S. electricity consumption (measured in megawatt-hours) using historical data from the Energy Information Administration spanning 2010-2025.

Two forecast scenarios are provided:
- **Conservative**: Projects more moderate growth patterns
- **Growth**: Assumes continuation of recent upward trends

Both models analyze historical patterns to predict future electricity demand. Use the slider to adjust how far into the future you'd like to see projections.
""")

# ----------------------------
# Calculate Test Set Accuracy
# ----------------------------
# Get test predictions for Model 1
test_pred_scaled_1 = model_1.get_forecast(steps=len(test_data)).predicted_mean
test_pred_1_unscaled = scaler_1.inverse_transform(test_pred_scaled_1.values.reshape(-1, 1)).flatten()
mape_1 = np.mean(np.abs((test_data['sales'].values - test_pred_1_unscaled) / test_data['sales'].values)) * 100
accuracy_1 = 100 - mape_1

# Get test predictions for Model 2
test_pred_scaled_2 = model_2.get_forecast(steps=len(test_data)).predicted_mean
test_pred_2_unscaled = scaler_2.inverse_transform(test_pred_scaled_2.values.reshape(-1, 1)).flatten()
mape_2 = np.mean(np.abs((test_data['sales'].values - test_pred_2_unscaled) / test_data['sales'].values)) * 100
accuracy_2 = 100 - mape_2

# ----------------------------
# Generate Future Forecasts (beyond test set)
# ----------------------------
# Model 1 forecast
future_pred_scaled_1 = model_1.get_forecast(steps=n_future).predicted_mean
future_pred_1 = scaler_1.inverse_transform(future_pred_scaled_1.values.reshape(-1, 1)).flatten()

# Model 2 forecast
future_pred_scaled_2 = model_2.get_forecast(steps=n_future).predicted_mean
future_pred_2 = scaler_2.inverse_transform(future_pred_scaled_2.values.reshape(-1, 1)).flatten()

last_date = energy_data.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=n_future,
    freq='MS'
)

# Calculate metrics
current_sales = energy_data['sales'].iloc[-1]
forecast_avg_1 = future_pred_1.mean()
forecast_avg_2 = future_pred_2.mean()
pct_change_1 = ((forecast_avg_1 - current_sales) / current_sales) * 100
pct_change_2 = ((forecast_avg_2 - current_sales) / current_sales) * 100

# ----------------------------
# KPI Metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Monthly Sales",
        f"{current_sales:,.0f} MWh",
        f"{last_date.strftime('%b %Y')}"
    )

with col2:
    st.metric(
        "Conservative Forecast Avg",
        f"{forecast_avg_1:,.0f} MWh",
        f"{pct_change_1:+.1f}%"
    )

with col3:
    st.metric(
        "Growth Forecast Avg",
        f"{forecast_avg_2:,.0f} MWh",
        f"{pct_change_2:+.1f}%"
    )

with col4:
    st.metric(
        "Forecast Period",
        f"{n_future} months",
        f"Through {future_dates[-1].strftime('%b %Y')}"
    )

st.markdown("---")

# ----------------------------
# Scenario Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Both Scenarios", "📉 Conservative Model", "📈 Growth Model"])

with tab1:
    st.subheader("Comparison: Conservative vs Growth Forecasts")
    
    fig_both = go.Figure()
    
    # Historical sales
    fig_both.add_trace(go.Scatter(
        x=energy_data.index,
        y=energy_data['sales'],
        mode='lines',
        line=dict(color='#000000', width=1),
        name='Historical Sales',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    # Conservative forecast
    fig_both.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred_1,
        mode='lines',
        line=dict(color='#dc3545', width=1),
        name='Conservative Forecast',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    # Growth forecast
    fig_both.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred_2,
        mode='lines',
        line=dict(color='#2ca02c', width=1),
        name='Growth Forecast',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    fig_both.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales (Megawatt Hours)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_both.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig_both.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    st.plotly_chart(fig_both, use_container_width=True)

with tab2:
    st.subheader("📉 Conservative Model Forecast")
    
    fig_1 = go.Figure()
    
    # Historical sales
    fig_1.add_trace(go.Scatter(
        x=energy_data.index,
        y=energy_data['sales'],
        mode='lines',
        line=dict(color='#000000', width=1),
        name='Historical Sales',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    # Forecast
    fig_1.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred_1,
        mode='lines',
        line=dict(color='#dc3545', width=1),
        name=f'Forecast ({n_future} months)',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    fig_1.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales (Megawatt Hours)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_1.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig_1.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    st.plotly_chart(fig_1, use_container_width=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Forecast Details")
        
        # Model accuracy metric
        st.metric(
            "Test Set Accuracy",
            f"{accuracy_1:.1f}%",
            help="Accuracy on held-out 2023+ data (100 - MAPE)"
        )
        
        forecast_df_1 = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales (MWh)': future_pred_1
        })
        forecast_df_1['Date'] = forecast_df_1['Date'].dt.strftime('%b %Y')
        forecast_df_1['Forecasted Sales (MWh)'] = forecast_df_1['Forecasted Sales (MWh)'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            forecast_df_1,
            hide_index=True,
            use_container_width=True,
            height=350
        )
    
    with col_right:
        st.subheader("📉 Recent Historical Trend")
        
        recent_data = energy_data.tail(24)
        
        fig_recent_1 = go.Figure()
        fig_recent_1.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['sales'],
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=1),
            marker=dict(size=6),
            name='Recent Sales',
            hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
        ))
        
        fig_recent_1.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales (MWh)",
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        fig_recent_1.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig_recent_1.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        st.plotly_chart(fig_recent_1, use_container_width=True)

with tab3:
    st.subheader("📈 Growth Model Forecast")
    
    fig_2 = go.Figure()
    
    # Historical sales
    fig_2.add_trace(go.Scatter(
        x=energy_data.index,
        y=energy_data['sales'],
        mode='lines',
        line=dict(color='#000000', width=1),
        name='Historical Sales',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    # Forecast
    fig_2.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred_2,
        mode='lines',
        line=dict(color='#2ca02c', width=1),
        name=f'Forecast ({n_future} months)',
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
    ))
    
    fig_2.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales (Megawatt Hours)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig_2.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig_2.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    st.plotly_chart(fig_2, use_container_width=True)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Forecast Details")
        
        # Model accuracy metric
        st.metric(
            "Test Set Accuracy",
            f"{accuracy_2:.1f}%",
            help="Accuracy on held-out 2023+ data (100 - MAPE)"
        )
        
        forecast_df_2 = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Sales (MWh)': future_pred_2
        })
        forecast_df_2['Date'] = forecast_df_2['Date'].dt.strftime('%b %Y')
        forecast_df_2['Forecasted Sales (MWh)'] = forecast_df_2['Forecasted Sales (MWh)'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(
            forecast_df_2,
            hide_index=True,
            use_container_width=True,
            height=350
        )
    
    with col_right:
        st.subheader("📉 Recent Historical Trend")
        
        recent_data = energy_data.tail(24)
        
        fig_recent_2 = go.Figure()
        fig_recent_2.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['sales'],
            mode='lines+markers',
            line=dict(color='#2ca02c', width=1),
            marker=dict(size=6),
            name='Recent Sales',
            hovertemplate='%{x|%b %Y}<br>%{y:,.0f} MWh<extra></extra>'
        ))
        
        fig_recent_2.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales (MWh)",
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        fig_recent_2.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig_recent_2.update_yaxes(showgrid=True, gridcolor='lightgray')
        
        st.plotly_chart(fig_recent_2, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
**Data Source:** U.S. Energy Information Administration (EIA)  
**Models:** Two SARIMA configurations trained on data through 2022, validated on 2023+ data  
**Note:** Forecasts are statistical projections and should be interpreted with appropriate uncertainty bounds.
""")