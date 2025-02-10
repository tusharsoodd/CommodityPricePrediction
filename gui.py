import streamlit as st
import pandas as pd
from schwartzSmith import SchwartzSmithModel, DataFetcher
from visualization_utils import (
    plot_price_predictions, plot_diagnostic_charts, print_diagnostics,
    _plot_prediction_errors, _plot_error_distribution, _plot_qq
)
import numpy as np
import time
import matplotlib.pyplot as plt

# Hide streamlit error messages
st.set_option('client.showErrorDetails', False)


# Create navigation links in the sidebar
st.sidebar.title("Navigation")
pages = {
    "Analysis": "Analysis",
    "About this project": "About"
}

# Get the current page from URL parameters
page = st.experimental_get_query_params().get("page", ["Analysis"])[0]

# Display navigation links
for page_name, page_title in pages.items():
    if st.sidebar.button(page_name):
        st.experimental_set_query_params(page=page_name)
        page = page_name

if page == "Analysis":
    st.title('Commodity Price Analysis with Schwartz-Smith Model')
    
    st.markdown("""
    This tool outputs:
    
    - Price predictions for the next period
    - Historical price decomposition showing short-term vs long-term trends
    - Model accuracy metrics and error analysis
    - Interactive charts comparing predicted vs actual prices
    - Real-time data for WTI Crude Oil, Brent Crude Oil, Natural Gas and Gasoline
    - Data intervals ranging from 1-minute to daily timeframes
    """)

    # Sidebar for user inputs
    st.sidebar.header('Configuration')

    # Commodity selection
    commodity_map = {
        'WTI Crude Oil': 'CL=F',
        'Brent Crude Oil': 'BZ=F', 
        'Natural Gas': 'NG=F',
        'Gasoline': 'RB=F'
    }
    selected_commodity = st.sidebar.selectbox(
        'Select Commodity',
        list(commodity_map.keys())
    )

    # Interval selection to match DataFetcher and SchwartzSmithModel defaults
    interval = st.sidebar.selectbox(
        'Select Interval',
        ['1m', '5m', '15m', '30m', '60m', '1d'],
        index=0
    )

    # Define valid periods for each interval
    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
        # Intraday data only available for last 7 days
        period = st.sidebar.selectbox(
            'Select Time Period',
            ['7d'],
            index=0
        )
    elif interval in ['1d']:
        # Daily data can go back further
        period = st.sidebar.selectbox(
            'Select Time Period',
            ['7d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            index=0
        )
    elif interval in ['1wk', '1mo']:
        # Weekly and monthly data best for longer periods
        period = st.sidebar.selectbox(
            'Select Time Period',
            ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            index=0
        )

    # Run button
    if st.sidebar.button('Run Analysis'):
        try:
            st.info('Starting analysis...')
            
            # Fetch data with selected parameters
            data_fetcher = DataFetcher(period=period, interval=interval)
            with st.spinner('Fetching price data...'):
                price_data = data_fetcher.fetch_data(commodity_map[selected_commodity])
            st.success('Price data fetched successfully!')

            # Initialize model with matching interval
            model = SchwartzSmithModel(interval=interval)
            
            st.info('Training new model...')
            with st.spinner('Processing...'):
                # Default parameters for training
                default_params = [0.5, 0.5, 0.5, 0.5, 0.0]  # sigma_chi, sigma_xi, sigma_seasonal, kappa, rho
                
                # Time the model training
                start_time = time.time()
                _, prices, predictions_ss, train_idx, diagnostics, next_forecast = model.fit_predict(
                    price_data,
                    default_params
                )
                training_time = time.time() - start_time
                st.success('Model trained successfully!')
            
            # Display results
            st.header('Results')

            # Dataset Information
            st.subheader('Dataset Information')
            total_samples = len(prices)
            train_samples = train_idx
            test_samples = total_samples - train_idx
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{total_samples}")
            with col2:
                st.metric("Training Samples", f"{train_samples}")
            with col3:
                st.metric("Testing Samples", f"{test_samples}")
            
            st.metric("Training Time", f"{training_time:.2f} seconds")
            
            # Next Price Forecast
            st.subheader('Next Price Forecast')
            forecast_period = {
                '1m': 'minute',
                '5m': '5 minutes',
                '15m': '15 minutes',
                '30m': '30 minutes',
                '60m': 'hour',
                '1d': 'day'
            }
            
            # Calculate percent change
            current_price = prices.values[-1]
            percent_change = ((next_forecast - current_price) / current_price) * 100
            change_direction = "increase" if percent_change > 0 else "decrease"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Forecasted Price for next {forecast_period[interval]}", 
                    f"${next_forecast:.2f}"
                )
            with col2:
                st.metric(
                    f"Predicted {change_direction}",
                    f"{abs(percent_change):.2f}%",
                    delta=f"{percent_change:+.2f}%"
                )
            
            # Metrics with interpretations
            st.subheader('Model Performance Metrics')
            train_r2, test_r2, train_mse, test_mse, train_mae, test_mae = diagnostics
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training R² Score", f"{train_r2:.4f}")
                st.markdown("""
                *R² Score indicates how well the model fits the training data. A score of 1.0 indicates perfect prediction, 
                while 0.0 indicates the model performs no better than a horizontal line.*
                """)
                
                st.metric("Training MSE", f"{train_mse:.4f}")
                st.markdown("""
                *Mean Squared Error (MSE) measures the average squared difference between predicted and actual values. 
                Lower values indicate better performance. MSE penalizes larger errors more heavily.*
                """)
                
                st.metric("Training MAE", f"$ {train_mae:.4f}")
                st.markdown("""
                *Mean Absolute Error (MAE) measures the average absolute difference between predicted and actual values. 
                It's more interpretable than MSE as it's in the same units as the original data.*
                """)
            
            with col2:
                st.metric("Testing R² Score", f"{test_r2:.4f}")
                if test_r2 > 0.8:
                    st.success("Strong predictive performance on unseen data")
                elif test_r2 > 0.6:
                    st.info("Moderate predictive performance on unseen data")
                else:
                    st.warning("Model may need improvement for better generalization")
                    
                st.metric("Testing MSE", f"{test_mse:.4f}")
                if test_mse < train_mse * 1.2:
                    st.success("Model generalizes well - similar error on test data")
                else:
                    st.warning("Model may be overfitting - higher error on test data")
                    
                st.metric("Testing MAE", f"$ {test_mae:.4f}")
                st.markdown(f"""
                *On average, predictions deviate by {test_mae:.4f} units from actual values on unseen data*
                """)
            
            # Plots
            st.subheader('Price Predictions')
            fig_price = plot_price_predictions(prices, train_idx, predictions_ss)
            st.pyplot(fig_price)

            st.subheader('Diagnostic Charts')
            fig_diag = plot_diagnostic_charts(prices, train_idx,  model.filtered_params, predictions_ss)
            st.pyplot(fig_diag)

            # Close all figures to free up memory
            plt.close('all')
            # Additional Error Analysis
            st.subheader('Error Analysis')
            
            # Calculate errors
            errors = prices - predictions_ss

            # Prediction Errors Plot
            fig_errors, ax = plt.subplots(figsize=(12, 6))
            _plot_prediction_errors(ax, prices, train_idx, errors)
            st.pyplot(fig_errors)
            plt.close()

            # Error Distribution
            fig_dist, ax = plt.subplots(figsize=(12, 6))
            _plot_error_distribution(ax, errors, train_idx)
            st.pyplot(fig_dist)
            plt.close()

            # Q-Q Plot
            fig_qq, ax = plt.subplots(figsize=(12, 6))
            _plot_qq(ax, errors[~np.isnan(errors)])
            st.pyplot(fig_qq)
            plt.close()

        except Exception as e:
            st.error("An error occurred during analysis. Please try again.")
            st.stop()

    else:
        st.write('Please select a commodity in the sidebar and click "Run Analysis" to start.')

elif page == "About this project":
    try:
        # Read and display the blog content
        with open('blog.txt', 'r', encoding='utf-8') as file:
            blog_content = file.read()
        
        st.markdown(blog_content)
    except Exception as e:
        st.error("Unable to load blog content. Please try again later.")
        st.stop()
