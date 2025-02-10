# Mathematical Framework for Price Decomposition and State Estimation

This repository contains the implementation of a mathematical framework used for modeling the log spot price of a commodity, as well as the estimation of underlying state variables using an Extended Kalman Filter (EKF). The framework models the dynamics of price decomposition into short-term, long-term, and seasonal components and employs state-space models for recursive estimation.

## Overview

The model consists of the following main components:

1. **Price Decomposition**: The log spot price is decomposed into a short-term mean-reverting component, a long-term equilibrium level, and a seasonal component.
2. **State Evolution**: The state variables follow certain stochastic differential equations (SDEs) to model their evolution over time.
3. **State Space Form**: A discretized system is defined for efficient computation, with a measurement equation for incorporating observed data.
4. **Extended Kalman Filter (EKF)**: A recursive algorithm to predict and update the state estimates based on noisy observations.

## Mathematical Framework

### 1. Price Decomposition

The log spot price of a commodity is modeled as:

$$
\ln(S_t) = \chi_t + \xi_t + s_t
$$

where:
- $\chi_t$: short-term mean-reverting component
- $\xi_t$: long-term equilibrium level
- $s_t$: seasonal component

### 2. State Evolution

The state dynamics follow:

$$
d\chi_t = -\kappa\chi_t \, dt + \sigma_\chi \, dW_\chi
$$

$$
d\xi_t = \mu_\xi \, dt + \sigma_\xi \, dW_\xi
$$

$$
dW_\chi dW_\xi = \rho \, dt
$$

### 3. State Space Form

Discretized system:

$$
X_t = \begin{pmatrix} \chi_t \\ \xi_t \\ s_t \end{pmatrix}
$$

$$
X_{t+1} = \begin{pmatrix} e^{-\kappa\Delta t} & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} X_t + w_t
$$

Measurement equation:

$$
y_t = \begin{pmatrix} 1 & 1 & 1 \end{pmatrix} X_t + v_t
$$

### 4. Estimation via Extended Kalman Filter

Recursive estimation using EKF:

- **Predict**: Update predicted state and covariance
- **Update**: Correct the predicted state using the measurement

## Installation

To use this framework, clone the repository:

