# MS CCC-GARCH Model Python Module

## Description

This Python module, designed as part of my Master's thesis "Modelling Commodity Futures Volatility" at University of Kiel, implements the Multivariate Markov Switching Constant Conditional Correlation GARCH (MS CCC-GARCH) model by Haas and Liu (2018) for n-variate case, supporting multivariate normal, student-t, and skewed-t distributions. It facilitates the estimation of parameters for different MS CCC-GARCH configurations via MLE, including options for regime-mean and autoregressive components.

## Results replication

The main results described in the thesis are available in the `thesis.ipynb` Jupyter notebook. To execute all the code cells sequentially, click on "Run All" in the notebook interface. This will process each cell from start to finish, reproducing the results as documented in the thesis.

## Installation

The code is written and tested in Python 3.9.19. To use this module, ensure you have Python and the necessary libraries installed:

```bash
pip install -r requirements.txt
```

## Usage

```bash
from ms_ccc_garch_2 import mgarch 

# Initialize the MGARCH model
model = mgarch(ret, dist='norm', regime_mean=False, ar=False, regime_ar=False)

# Fit the model
result = model.fit()

# Retrieve smoothed probabilities and parameter estimates
xsi, chi, condcorr, est, h1, h2, loglik = model.get_smoothed_prob(result.x)
```


## References
1. [Markus Haas & Ji-Chun Liu, "A Multivariate Regime-switching GARCH Model" Code Ocean.](https://codeocean.com/capsule/9016375/tree/v1)
Implementation of MS CCC-GARCH model estimation and forecasting for bivariate case
2. [Jeffrey Pollock, "Multivariate normal covariance matrices and the cholesky decomposition" Github.](https://jeffpollock9.github.io/multivariate-normal-cholesky/)
On efficient estimation of multivariate distributions using cholesky decomposition
3. [Archakov, I. & Reinhard Hansen, P., 2021. "A New Parametrization of Correlation Matrices." _Econometrica_, Volume 89, pp. 1699-1715.](https://homepage.univie.ac.at/ilya.archakov/docs/CorNoteWebAppendix.pdf)
Algorithm for constrained optimization of PSD correlation matrix
4. [Weibel, M., Breyman, W., & Lüthi, D. (2023). ghyp: A package on generalized hyperbolic distributions_. Retrieved from [CRAN]](https://cran.r-project.org/web/packages/ghyp/vignettes/Generalized_Hyperbolic_Distribution.pdf)
Implementation of multivariate Generalized Hyperbolic distribution in R