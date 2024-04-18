# MS CCC-GARCH Model Python Module

## Description

This Python module implements the Multivariate Markov Switching Constant Conditional Correlation GARCH (MS CCC-GARCH) model by Haas and Liu (2018), supporting multivariate normal, student-t, and skewed-t distributions. It facilitates the estimation of parameters for different MS CCC-GARCH configurations, including options for regime-mean, autoregressive components, and distribution choices. 

## Features

- Support for various distributions: normal, student-t, and skewed-t.
- Functionality for both regime mean and autoregressive components.
- Robust error handling and parameter initialization.
- Detailed likelihood computation and optimization.

## Installation

To use this module, ensure you have Python and the necessary libraries installed:

```bash
pip install -r requirements.txt


Usage
Import the module and initialize the mgarch class with your data.
Configure the model parameters and distribution.
Use the fit method to estimate the parameters.
Access various attributes and methods for further analysis, such as smoothed probabilities and log-likelihood calculations.

from ms_ccc_garch_2 import mgarch 

# Initialize the MGARCH model
model = mgarch(ret, dist='norm', regime_mean=False, ar=False, regime_ar=False)

# Fit the model
result = model.fit()

# Retrieve smoothed probabilities and parameter estimates
xsi, chi, condcorr, est, h1, h2, loglik = model.get_smoothed_prob(result.x)
