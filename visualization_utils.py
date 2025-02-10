import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np


def plot_price_predictions(openPrice, train_idx, predictions_ss):
    """Plot the training data, test data, and model predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(openPrice.index[:train_idx], openPrice.values[:train_idx], label='Training Data', alpha=0.7)
    ax.plot(openPrice.index[train_idx:], openPrice.values[train_idx:], label='Test Data', alpha=0.7)
    ax.plot(openPrice.index, predictions_ss, label='Model Predictions', alpha=0.7)
    ax.axvline(x=openPrice.index[train_idx], color='r', linestyle=':', label='Train/Test Split')
    ax.set_title('Schwartz-Smith Model with Optimized Parameters')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    return fig


def plot_diagnostic_charts(openPrice, train_idx, filtered_params, predictions_ss):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate errors
    errors = openPrice.values - predictions_ss
    
    # Plot 1: Factor contributions
    _plot_factor_contributions(axes[0, 0], openPrice, train_idx, filtered_params)
    
    # Plot 2: Prediction errors
    _plot_prediction_errors(axes[0, 1], openPrice, train_idx, errors)
    
    # Plot 3: Error distribution
    _plot_error_distribution(axes[1, 0], errors, train_idx)
    
    # Plot 4: Q-Q plot of errors
    _plot_qq(axes[1, 1], errors[~np.isnan(errors)])
    
    plt.tight_layout()
    return fig


def _plot_factor_contributions(ax, openPrice, train_idx, filtered_params):
    """Helper function to plot factor contributions."""
    ax.plot(openPrice.index, filtered_params[:, 0], label='Short-term')
    ax.plot(openPrice.index, filtered_params[:, 1], label='Long-term')
    ax.plot(openPrice.index, filtered_params[:, 2], label='Seasonal')
    ax.axvline(x=openPrice.index[train_idx], color='r', linestyle=':', label='Train/Test Split')
    ax.set_title('Factor Contributions')
    ax.legend()



def _plot_prediction_errors(ax, openPrice, train_idx, errors):
    """Helper function to plot prediction errors."""
    ax.plot(openPrice.index, errors)
    ax.axvline(x=openPrice.index[train_idx], color='r', linestyle=':', label='Train/Test Split')
    ax.set_title('Prediction Errors')
    ax.legend()

def _plot_error_distribution(ax, errors, train_idx):
    """Helper function to plot error distribution."""
    train_errors = errors[:train_idx]
    test_errors = errors[train_idx:]
    
    # Filter out NaN values
    train_errors = train_errors[~np.isnan(train_errors)]
    test_errors = test_errors[~np.isnan(test_errors)]
    
    if len(train_errors) > 0 and len(test_errors) > 0:
        ax.hist(train_errors, bins=50, alpha=0.5, label='Train Errors')
        ax.hist(test_errors, bins=50, alpha=0.5, label='Test Errors')
        ax.set_title('Error Distribution')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No valid error data to plot', 
                horizontalalignment='center',
                verticalalignment='center')

def _plot_qq(ax, errors):
    """Helper function to create Q-Q plot."""
    if len(errors) > 0:
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Errors')
    else:
        ax.text(0.5, 0.5, 'No valid error data for Q-Q plot', 
                horizontalalignment='center',
                verticalalignment='center')

def print_diagnostics(train_r2, test_r2, train_mse, test_mse, train_mae, test_mae):
    """Print model performance metrics"""
    print("\nModel Performance Metrics:")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Testing MAE: {test_mae:.4f}")

def _plot_residuals(pricesWithIndex, predictions, train_idx):
    """Plot the residuals"""
    residuals = pricesWithIndex.values - predictions
    
    plt.plot(pricesWithIndex.index, residuals, label='Residuals')
    plt.axvline(x=pricesWithIndex.index[train_idx], color='r', linestyle='--', alpha=0.5)
    
    plt.title('Model Residuals')
    plt.xlabel('Time')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.grid(True)