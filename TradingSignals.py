class TradingSignals:
    def __init__(self, schwartz_smith_model, confidence_level=1.96):  # 95% confidence interval
        self.model = schwartz_smith_model
        self.confidence_level = confidence_level

    def trend_based_signals(self, price_data):
        """
        Generate trading signals based on long-term factor from Schwartz-Smith model
        Returns: Array of signals (1 for Buy, -1 for Sell, 0 for Hold)
        """
        signals = np.zeros(len(price_data))
        long_term_factor = self.model.filtered_params[:, 1]  # Long-term factor from SS model
        
        for i in range(len(price_data)):
            if long_term_factor[i] > price_data[i]:
                signals[i] = 1  # Buy signal - undervalued
            elif long_term_factor[i] < price_data[i]:
                signals[i] = -1  # Sell signal - overvalued
        return signals

    def mean_reversion_signals(self, threshold=1.0):
        """
        Generate trading signals based on short-term deviations from SS model
        threshold: Number of standard deviations to trigger signal
        Returns: Array of signals (1 for Buy, -1 for Sell, 0 for Hold)
        """
        signals = np.zeros(len(self.model.filtered_params))
        short_term_factor = self.model.filtered_params[:, 0]  # Short-term factor from SS model
        
        # Calculate standard deviation of short-term factor
        std_dev = np.std(short_term_factor)
        
        for i in range(len(short_term_factor)):
            if short_term_factor[i] > threshold * std_dev:
                signals[i] = -1  # Sell signal - expect downward correction
            elif short_term_factor[i] < -threshold * std_dev:
                signals[i] = 1  # Buy signal - expect upward correction
        return signals

    def kalman_filter_signals(self, price_data):
        """
        Generate trading signals based on Kalman Filter confidence bands from SS model
        Returns: Array of signals (1 for Buy, -1 for Sell, 0 for Hold)
        """
        signals = np.zeros(len(price_data))
        filtered_estimate = np.sum(self.model.filtered_params, axis=1)  # Combined state estimate
        uncertainty = np.diag(self.model._init_ekf(price_data[0]).P)  # Get uncertainty from KF
        
        for i in range(len(price_data)):
            upper_band = filtered_estimate[i] + self.confidence_level * np.sqrt(uncertainty[1])  # Using long-term uncertainty
            lower_band = filtered_estimate[i] - self.confidence_level * np.sqrt(uncertainty[1])
            
            if price_data[i] > upper_band:
                signals[i] = -1  # Sell signal - overbought
            elif price_data[i] < lower_band:
                signals[i] = 1  # Buy signal - oversold
        return signals

    def combined_strategy(self, price_data):
        """
        Combined trading strategy integrating Schwartz-Smith model components
        
        Strategy logic:
        1. Uses model's decomposition of price into short-term and long-term components
        2. Weights signals based on model uncertainty
        3. Incorporates seasonal component for timing adjustments
        
        Returns: Array of signals (1 for Buy, -1 for Sell, 0 for Hold)
        """
        signals = np.zeros(len(price_data))
        
        # Get individual signals
        trend_signals = self.trend_based_signals(price_data)
        reversion_signals = self.mean_reversion_signals()
        kalman_signals = self.kalman_filter_signals(price_data)
        
        # Get seasonal component from SS model
        seasonal_factor = self.model.filtered_params[:, 2]
        
        # Get model uncertainty
        uncertainty = np.diag(self.model._init_ekf(price_data[0]).P)
        uncertainty_ratio = uncertainty[0] / (uncertainty[0] + uncertainty[1])  # Relative uncertainty between factors
        
        for i in range(len(price_data)):
            # Weight signals based on uncertainty and seasonal factors
            trend_weight = 1 - uncertainty_ratio  # Higher weight when long-term more certain
            reversion_weight = uncertainty_ratio  # Higher weight when short-term more certain
            
            # Adjust weights based on seasonal component
            seasonal_adjustment = np.abs(seasonal_factor[i]) / np.max(np.abs(seasonal_factor))
            
            # Combine signals with dynamic weights
            combined_signal = (
                trend_signals[i] * trend_weight * (1 - seasonal_adjustment) +
                reversion_signals[i] * reversion_weight * seasonal_adjustment
            )
            
            # Use Kalman signals as confirmation
            if kalman_signals[i] != 0:
                combined_signal *= np.sign(kalman_signals[i])
            
            # Convert to final trading decision
            if combined_signal > 0.5:
                signals[i] = 1
            elif combined_signal < -0.5:
                signals[i] = -1
                
        return signals

    def generate_all_signals(self, price_data):
        """
        Generate all trading signals using the Schwartz-Smith model components
        Returns: Dictionary of signal arrays
        """
        return {
            'trend_signals': self.trend_based_signals(price_data),
            'reversion_signals': self.mean_reversion_signals(),
            'kalman_signals': self.kalman_filter_signals(price_data),
            'combined_signals': self.combined_strategy(price_data)
        }

