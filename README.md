# Schwartz-Smith Commodity Price Analysis Tool

An implementation of the Schwartz-Smith model for commodity price analysis and forecasting, featuring Extended Kalman Filter state estimation and automated parameter optimization.

## Overview

This tool implements the Schwartz-Smith two-factor model with an additional seasonal component for commodity price analysis. It combines:
- Extended Kalman Filter for state estimation
- Bayesian optimization for parameter tuning
- Trading signal generation
- Interactive visualization through Streamlit

## Features

- **Advanced Price Modeling**
  - Three-factor decomposition (short-term, long-term, seasonal)
  - Extended Kalman Filter state estimation
  - Automated parameter optimization

- **Trading Signals**
  - Model-based buy/sell signal generation
  - Performance metrics calculation
  - Risk analysis

- **Visualization**
  - Interactive price predictions
  - Factor contribution analysis
  - Error distribution analysis
  - Model diagnostics

## Installation

1. Clone the repository: 
```
bash
git clone https://github.com/yourusername/schwartz-smith-analysis.git
cd schwartz-smith-analysis
```

2. Install dependencies:
```
bash
pip install -r requirements.txt
```

## Usage

1. Launch the Streamlit interface:
```
bash
streamlit run gui.py
```


## Project Structure
```
├── gui.py # Streamlit interface\n
├── schwartzSmith.py # Core model implementation
├── TradingSignals.py # Trading signal generation
├── visualization_utils.py # Plotting utilities
└── requirements.txt # Project dependencies
```

## Model Components

### Schwartz-Smith Model
- Short-term mean-reverting component
- Long-term random walk component
- Seasonal factor
- Extended Kalman Filter implementation

### Trading Signals
- Model-based signal generation
- Performance metrics
- Risk analysis

### Visualization
- Price predictions
- Factor decomposition
- Error analysis
- Diagnostic plots

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.2.0
streamlit>=1.24.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
filterpy>=1.4.5
scipy>=1.7.0
plotly>=5.3.0
seaborn>=0.11.0
scikit-optimize>=0.9.0
```


