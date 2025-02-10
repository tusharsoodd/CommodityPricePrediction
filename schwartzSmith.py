import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from skopt import gp_minimize
from skopt.space import Real
import json
import os
import yfinance as yf
import pandas as pd
from visualization_utils import plot_price_predictions, plot_diagnostic_charts, print_diagnostics


class DataFetcher:
    def __init__(self, period="7d", interval="1m"):
        """
        Initialize DataFetcher
        Args:
            period: Time period to fetch (e.g. "7d", "1mo", "1y")
            interval: Data interval ("1m", "5m", "15m", "30m", "60m", "1d")
        """
        self.period = period
        self.interval = interval
        
    def fetch_data(self, ticker_symbol):
        """
        Fetch historical price data for a given ticker symbol
        Returns: pandas.Series with datetime index and float values
        """
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period=self.period, interval=self.interval)
        return pd.Series(history["Open"], index=history.index)
        
    def fetch_multiple(self, ticker_symbols):
        """
        Fetch data for multiple ticker symbols
        Args:
            ticker_symbols: dict of {symbol: name} pairs
        Returns: dict of {symbol: pd.Series} pairs
        """
        return {symbol: self.fetch_data(symbol) for symbol in ticker_symbols}


class SchwartzSmithModel:
    def __init__(self, interval="1m", commodity=None):
        """
        Initialize model with configurable time interval
        Args:
            interval: Data interval ("1m", "5m", "15m", "30m", "60m", "1d")
        """
        # Map intervals to dt values
        interval_map = {
            "1m": 1/(252*390),  
            "5m": 5/(252*390),
            "15m": 15/(252*390),
            "30m": 30/(252*390),
            "60m": 60/(252*390),
            "1d": 1/252
        }
        self.dt = interval_map.get(interval, 1/(252*390))
        self.commodity = commodity
        
        self.filtered_params = None  # Shape: (n_samples, 3)
        self.filtered_prices = None  # Shape: (n_samples,)
        self.train_idx = None
        self.next_forecast = None  # Store next period forecast
        
    def _init_ekf(self, initial_price):
        """Initialize Extended Kalman Filter"""
        ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        ekf.x = np.array([0., initial_price, 0.])
        ekf.P = np.eye(3) * 0.1
        ekf.R = np.array([[0.05]])
        return ekf
    
    def _setup_process_noise(self, ekf, params):
        """
        Setup process noise covariance matrix
        Args:
            params: List [sigma_chi, sigma_xi, sigma_seasonal, kappa, rho]
        """
        sigma_chi, sigma_xi, sigma_seasonal, _, rho = params
        ekf.Q = np.zeros((3, 3))
        ekf.Q[0,0] = sigma_chi**2 * self.dt
        ekf.Q[1,1] = sigma_xi**2 * self.dt
        ekf.Q[2,2] = sigma_seasonal**2 * self.dt
        ekf.Q[0,1] = rho * sigma_chi * sigma_xi * self.dt
        ekf.Q[1,0] = ekf.Q[0,1]
        return ekf
    
    def _state_transition(self, x, kappa):
        """State transition function with seasonality"""
        chi, xi, seasonal = x
        chi_new = chi * np.exp(-kappa * self.dt)
        xi_new = xi
        seasonal_new = seasonal * np.cos(2 * np.pi * self.dt)
        return np.array([chi_new, xi_new, seasonal_new])
    
    @staticmethod
    def _measurement(x):
        """Measurement function with seasonality"""
        return np.array([x[0] + x[1] + x[2]])
    
    @staticmethod
    def _HJacobian(x):
        """Measurement Jacobian matrix"""
        return np.array([[1.0, 1.0, 1.0]])    
    
    def fit_predict(self, pricesWithIndex, params, train_size=0.75):
        """
        Fit model and generate predictions
        Args:
            prices: pd.Series or np.array of prices
            params: List [sigma_chi, sigma_xi, sigma_seasonal, kappa, rho]
            train_size: Float between 0 and 1
        Returns:
            Tuple (objective_value, prices, predictions, train_idx, metrics)
        """
        prices = pricesWithIndex.values
        prices = np.asarray(prices)
        prices = np.maximum(prices, 1e-6)
        log_prices = np.log(prices)
        n = len(prices)
        
        self.train_idx = int(n * train_size)
        train_prices = prices[:self.train_idx]
        test_prices = prices[self.train_idx:]
        train_log_prices = log_prices[:self.train_idx]
        
        ekf = self._init_ekf(train_log_prices[0])
        ekf = self._setup_process_noise(ekf, params)
        
        self.filtered_params = np.zeros((n, 3))
        self.filtered_prices = np.zeros(n)
        self.filtered_prices[0] = train_prices[0]
        
        # Train the model
        self._train_model(ekf, train_log_prices)
        
        # Test the model using the trained state
        self._test_model(ekf, log_prices[self.train_idx:], params[3])
        
        # Generate next period forecast
        next_state = self._state_transition(ekf.x, params[3])
        self.next_forecast = np.exp(np.sum(next_state))
        
        metrics = self._calculate_metrics(prices, train_prices, test_prices)
        return (-metrics[1], pricesWithIndex, self.filtered_prices, self.train_idx, metrics, self.next_forecast)
    
    def _train_model(self, ekf, train_log_prices):
        """Training phase of the model"""
        for t in range(1, self.train_idx):
            ekf.x = self._state_transition(ekf.x, ekf.Q[3] if hasattr(ekf.Q, '3') else 0.5)
            ekf.P = ekf.P + ekf.Q
            
            log_price = train_log_prices[t]
            if np.isfinite(log_price):
                ekf.update(log_price, HJacobian=self._HJacobian, Hx=self._measurement)
            
            self.filtered_params[t] = ekf.x
            self.filtered_prices[t] = np.exp(np.sum(ekf.x))
    
    def _test_model(self, ekf, test_log_prices, kappa):
        """Testing phase using trained model state"""
        for t in range(len(test_log_prices)):
            # Make prediction before updating state
            ekf.x = self._state_transition(ekf.x, kappa)
            ekf.P = ekf.P + ekf.Q
            
            # Store prediction
            self.filtered_params[t + self.train_idx] = ekf.x
            self.filtered_prices[t + self.train_idx] = np.exp(np.sum(ekf.x))
            
            # Update state with actual observation
            log_price = test_log_prices[t]
            if np.isfinite(log_price):
                ekf.update(log_price, HJacobian=self._HJacobian, Hx=self._measurement)
    
    def _calculate_metrics(self, prices, train_prices, test_prices):
        """Calculate performance metrics"""
        train_r2 = r2_score(train_prices, self.filtered_prices[:self.train_idx])
        test_r2 = r2_score(test_prices, self.filtered_prices[self.train_idx:])
        train_mse = mean_squared_error(train_prices, self.filtered_prices[:self.train_idx])
        test_mse = mean_squared_error(test_prices, self.filtered_prices[self.train_idx:])
        train_mae = mean_absolute_error(train_prices, self.filtered_prices[:self.train_idx])
        test_mae = mean_absolute_error(test_prices, self.filtered_prices[self.train_idx:])
        return (train_r2, test_r2, train_mse, test_mse, train_mae, test_mae)


class ParameterOptimizer:
    def __init__(self, model, prices, params_file='ss_model_params.json'):
        """
        Initialize optimizer
        Args:
            model: SchwartzSmithModel instance
            prices: pd.Series or np.array of prices
            params_file: String path to save parameters
        """
        self.model = model
        self.prices = np.asarray(prices)
        self.params_file = params_file
        self.best_params = None
        self.best_value = float('inf')
        
    def _define_parameter_space(self):
        """Define the parameter space for optimization"""
        return [
            Real(0.1, 5.0, name='sigma_chi'),
            Real(0.01, 5.0, name='sigma_xi'),
            Real(0.01, 5.0, name='sigma_seasonal'),
            Real(0.1, 2.0, name='kappa'),
            Real(-0.9, 0.9, name='rho')
        ]
    
    def _load_previous_params(self):
        """Load previously saved parameters if they exist"""
        if os.path.exists(self.params_file):
            with open(self.params_file, 'r') as f:
                saved_data = json.load(f)
                self.best_params = [
                    saved_data['sigma_chi'],
                    saved_data['sigma_xi'],
                    saved_data['sigma_seasonal'],
                    saved_data['kappa'],
                    saved_data['rho']
                ]
                self.best_value = saved_data['objective_value']
                print("Loaded previous best parameters with value:", self.best_value)
    
    def _save_params(self, params, value):
        """Save parameters to file"""
        params_dict = {
            'sigma_chi': params[0],
            'sigma_xi': params[1],
            'sigma_seasonal': params[2],
            'kappa': params[3],
            'rho': params[4],
            'objective_value': value
        }
        with open(self.params_file, 'w') as f:
            json.dump(params_dict, f, indent=4)
    
    def optimize(self):
        """Run Bayesian optimization"""
        space = self._define_parameter_space()
        self._load_previous_params()
        
        def objective(params):
            value = self.model.fit_predict(self.prices, params)[0]
            
            if value < self.best_value:
                self.best_value = value
                self.best_params = params
                self._save_params(params, value)
                print(f"New best parameters found with value: {value}")
            
            return value
        
        print("Starting Bayesian optimization of parameters...")
        result = gp_minimize(
            objective,
            space,
            n_calls=2,
            n_random_starts=1,
            noise=0.4,
            verbose=True,
            x0=self.best_params
        )
        
        self._print_results(result.x)
        return result.x
    
    def _print_results(self, params):
        """Print optimization results"""
        print("\nOptimized Parameters:")
        print(f"sigma_chi: {params[0]:.3f}")
        print(f"sigma_xi: {params[1]:.3f}")
        print(f"sigma_seasonal: {params[2]:.3f}")
        print(f"kappa: {params[3]:.3f}")
        print(f"rho: {params[4]:.3f}")
