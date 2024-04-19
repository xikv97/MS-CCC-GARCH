import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2
import vartests
import statsmodels.api as sm
from scipy.stats import multivariate_normal,invgamma
from numpy import exp,diag,zeros,array,std,log,sqrt,prod


""" 
miscellaneous functions used for some statisticall tests and tables creation
"""
def bic(loglik,params,ret):
    T,d = ret.shape
    bic = -2*loglik + len(params)*np.log(T)
    return bic
def aic(loglik,params,ret):
    T,d = ret.shape
    aic = -2*loglik + len(params) * 2
    return aic

def return_lrt(loglik,params,ret):
    ll = np.round(loglik,0)
    l_aic = np.round(aic(loglik,params,ret),0)
    l_bic =  np.round(bic(loglik,params,ret),0)
    print(f'Loglik: {ll.item()}')
    print(f'AIC: {l_aic.item()}')
    print(f'BIC: {l_bic.item()}')

def ewma(epsilon, lam):
    T = epsilon.shape[0]
    Sigma = np.cov(epsilon, rowvar=False)
    rho = Sigma[1, 0] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])

    for t in range(1, T):
        Sigma = lam * Sigma + (1 - lam) * np.outer(epsilon[t - 1, :], epsilon[t - 1, :])
        rho_t = Sigma[1, 0] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
        rho = np.vstack((rho, rho_t))

    return rho

def hessian2(funfcn, b, d, *args):
    # Compute the hessian matrix of the 2nd partial derivative
    # for the likelihood function f evaluated at b


    del_del = d
    n = b.shape[0]
    ey = np.eye(n)
    f1 = np.zeros((n, n))
    f2 = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1):
            b_plus = b + del_del * (ey[:, i] + ey[:, j])
            b_minus = b + del_del * (ey[:, i] - ey[:, j])
            f1[i, j] = funfcn(b_plus, *args) - funfcn(b_minus, *args)
            f1[j, i] = f1[i, j]

            b_plus = b - del_del * (ey[:, i] - ey[:, j])
            b_minus = b - del_del * (ey[:, i] + ey[:, j])
            f2[i, j] = funfcn(b_plus, *args) - funfcn(b_minus, *args)
            f2[j, i] = f2[i, j]

    hessn = (f1 - f2) / (4 * del_del * del_del)
    return hessn

def calculate_statistics(returns):
    """
    Calculates statistical measures for each column in an array of returns.
    
    Args:
        returns (numpy.ndarray): Array of returns where each column represents a different asset.
    
    Returns:
        pandas.DataFrame: Table with columns 'Max', 'Mean', 'Median', 'Min', 'Std', 'Skew', 'Kurtosis', and 'Jarque-Bera'.
    """
    statistics = {
        'Max': np.max(returns, axis=0),
        'Mean': np.mean(returns, axis=0),
        'Median': np.median(returns, axis=0),
        'Min': np.min(returns, axis=0),
        'Std': np.std(returns, axis=0),
        'Skew': stats.skew(returns, axis=0),
        'Kurtosis': stats.kurtosis(returns, axis=0),
        'Jarque-Bera': [],
        'Jarque-Bera p-value': []
    }
    
    for col in returns.T:
        jb_value, jb_p_value = stats.jarque_bera(col)
        statistics['Jarque-Bera'].append(jb_value)
        statistics['Jarque-Bera p-value'].append(jb_p_value)
    
    return pd.DataFrame(statistics,index=['ttf', 'pow', 'eua'])

def std_errors(fun,params):
    Hnormsingle = hessian2(fun, params, 0.001)
    matrix = -np.linalg.inv(Hnormsingle) * np.eye(Hnormsingle.shape[0])
    # Compute the square root of the diagonal elements
    sqrt_diag = np.sqrt(np.diag(matrix))
    return sqrt_diag.round(3)

def LRT_test(loglik_1,loglik_2,params_1,params_2):
    LRT = 2 * (loglik_1 - loglik_2)
    p_value = 1 - chi2.cdf(LRT, params_1.shape[0] - params_2.shape[0])
    return {'LRT':LRT , 'p_value':p_value}

def kupiec_test_statistic(total_observations, exceedances, confidence_level):
    """
    Calculate the Kupiec test statistic for a given VaR model performance.

    Parameters:
    - total_observations: The total number of observations in the dataset.
    - exceedances: The number of times the loss exceeded the VaR estimate.
    - confidence_level: The confidence level used for the VaR estimate (e.g., 0.95 for 95%).

    Returns:
    - LR: The likelihood ratio (LR) test statistic.
    - p_value: The p-value from the chi-square distribution.
    """
    # Probability of exceedance under the model
    p = 1 - confidence_level
    n = total_observations
    x = exceedances

    # Expected number of exceedances
    np_expected = n * p

    # Likelihood ratio test statistic
    if x > 0 and x < n:  # Avoid division by zero or log of zero
        LR = -2 * np.log(((1-p)**(n-x) * p**x) / ((1-x/n)**(n-x) * (x/n)**x))
    elif x == 0:  # All in (1-p) term
        LR = -2 * np.log((1-p)**n / (1-x/n)**n)
    else:  # x == n, all in p term
        LR = -2 * np.log(p**n / (x/n)**n)

    # Calculate the p-value
    p_value = chi2.sf(LR, 1)  # 1 degree of freedom for Kupiec test

    return LR, p_value

def return_vartest_table(violations):

    df_ret = pd.DataFrame({
    'Violation ratio': vartests.failure_rate(violations)['failure rate'],
    'Kupiec': [kupiec_test_statistic(len(violations), sum(violations), 0.95)[1]],
    'Christoffersen': vartests.duration_test(violations, conf_level=0.95)['log-likelihood ratio test statistic'],
    }, index=[0])
    return df_ret.round(2)

def var_table(vio_list):
    df_ret = pd.DataFrame()
    for vio in vio_list:
        df = return_vartest_table(vio)
        df_ret = pd.concat([df_ret,df])
    return df_ret

def engle_colacito_table(list_ret):
    # Initialize matrices to store coefficients and p-values
    n = len(list_ret)
    coef_matrix = np.zeros((n, n))
    p_value_matrix = np.zeros((n, n))

    # Perform pairwise regressions and fill the matrices
    for i in range(n):
        for j in range(n):
            if i != j:
                # Compute the squared differences
                Y = np.square(list_ret[i]) - np.square(list_ret[j])
                X = np.ones(len(list_ret[j]))
                # Create and fit the model
                model = sm.OLS(Y, X)
                results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
                # Store the coefficient and p-value
                coef_matrix[i, j] = results.params[0]
                p_value_matrix[i, j] = results.pvalues[0]

    # Print matrices
    print("Coefficient Matrix:")
    print(coef_matrix *-1)
    print("\nP-Value Matrix:")
    print(p_value_matrix)

def skew_t_MCMC(mu,w1,w2,phi1,phi2,A1,A2,B1,B2,G,p,q,nu,P,R1,R2,sigma1init,sigma2init,pinit,gam,ret,d = 3,N = 2000,T=1):

    S = np.zeros(T)

    S[0] = np.argmax(pinit) + 1

    h1,h2 = zeros([d,T]),zeros([d,T])

    rt = zeros([d,T])

    h1[:,0],h2[:,0] = sigma1init,sigma2init

    #p1 = np.zeros(T + 1)
    #p2 = np.zeros(T + 1)

    #p1[0] = (1 - q) / (2 - p - q)
    #p2[0] = 1 - p1[0]

    eps = np.zeros([d,T+1])

    hi = np.zeros([d,T])

    hi[:,0] = h1[:,0]

    Rt = np.zeros([d,T,N])

    a = nu / 2  # Shape parameter for the inverse gamma, derived from nu
    scale = nu / 2
    
    for n in range(N):
        
        samples1 = multivariate_normal.rvs(np.zeros(d),np.eye(d), T+1)
        samples2 = multivariate_normal.rvs(np.zeros(d),np.eye(d), T+1)
        sample_inv_gamma = invgamma.rvs(a=a, scale=scale, size=T+1)

        L1 = np.linalg.cholesky((R1))
        L2 = np.linalg.cholesky((R2))

        eps_MC = eps[:,0]
        eps_MC =  eps_MC.T
        abs_eps_MC = abs(eps_MC)

        if S[0] == 1:
            rt[:,0] = mu + phi1 * (ret[-1,:] - mu) + gam * sample_inv_gamma[0] +  np.sqrt(sample_inv_gamma[0]) * diag(h1[:,0]) @ (L1)  @ samples1[0,:]
            hi[:,0] = h1[:,0]

        elif S[0] == 2:
            rt[:,0] = mu + phi1 * (ret[-1,:] - mu) + gam * sample_inv_gamma[0] +  np.sqrt(sample_inv_gamma[0]) * diag(h2[:,0]) @ (L2)  @ samples2[0,:]
            hi[:,0] = h2[:,0]
        
        eps[:,0] = rt[:,0] - mu + phi1 * (ret[-1,:] - mu) 

        for t in range(1, T):
            u = np.random.rand()
            
            eps_MC = eps[:,t-1]
            eps_MC =  eps_MC.T
            abs_eps_MC = abs(eps_MC)

            h1[:,t] = w1 + A1@abs_eps_MC - A1 @ G @ eps[:,t-1]  + B1 @ h1[:,t-1]
            h2[:,t] = w2 + A2@abs_eps_MC - A2 @ G @ eps[:,t-1]  + B2 @ h2[:,t-1]
                
            if S[t-1] == 1:
                if u < p:
                    S[t] = 1
                    rt[:,t] = mu + phi1 * (rt[:,t-1] - mu) + gam * sample_inv_gamma[t] +  np.sqrt(sample_inv_gamma[t]) * diag(h1[:,t]) @ (L1)  @ samples1[t,:]
                    hi[:,t] = h1[:,t]
                else:
                    S[t] = 2
                    rt[:,t] = mu + phi1 *(rt[:,t-1] - mu )+ gam * sample_inv_gamma[t] +  np.sqrt(sample_inv_gamma[t]) * diag(h2[:,t]) @ (L2)  @ samples2[t,:]
                    hi[:,t] = h2[:,t]
            elif S[t-1] == 2:
                if u < q:
                    S[t] = 2
                    rt[:,t] = mu + phi1 * (rt[:,t-1] - mu) + gam * sample_inv_gamma[t] +  np.sqrt(sample_inv_gamma[t]) * diag(h2[:,t]) @ (L2)  @ samples2[t,:]
                    hi[:,t] = h2[:,t]
                else:
                    S[t] = 1
                    rt[:,t] = mu + phi1 *( rt[:,t-1]-mu) + gam * sample_inv_gamma[t] +  np.sqrt(sample_inv_gamma[t]) * diag(h1[:,t]) @ (L1)  @ samples1[t,:]
                    hi[:,t] = h1[:,t]


            eps[:,t] = rt[:,t] - mu - phi1 * (rt[:,t-1] - mu)
                        
        Rt[:, :, n] = rt[:,:]
        
    return Rt

def calculate_variance(ER_list, A, E, h):
    """" 
    Recursion for multi step ahead covariance matrix forecast 
    Hlouskova (2009)
    """
    variance = 0 
    for i in range(1, h+1):
        variance +=  E.T @ ER_list[i-1] @ E
    for i in range(2, h+1):
        for j in range(1, i):
            variance += E.T @ np.linalg.matrix_power(A, i-j) @ ER_list[j-1] @ E
    for i in range(1, h):
        for j in range(i+1, h+1):
            variance += E.T @ ER_list[i-1] @ np.linalg.matrix_power(A, j-i).T @ E
    return variance


def calculate_maximum_drawdown(pnl_vector):
    """
    Calculate the maximum drawdown in a PnL vector.

    Parameters:
    - pnl_vector: A numpy array or list containing the PnL values.

    Returns:
    - max_drawdown: The maximum drawdown value.
    """
    # Convert pnl_vector to a numpy array if it isn't one already
    pnl_vector = np.array(pnl_vector)

    # Calculate the running maximum
    running_max = np.maximum.accumulate(pnl_vector)

    # Calculate drawdown
    drawdown = running_max - pnl_vector

    # Find maximum drawdown
    max_drawdown = np.max(drawdown)

    return max_drawdown


def return_portfolio_stats(pnl_arr):
    average_return = np.mean(pnl_arr[:])
    # Calculate standard deviation of daily return
    volatility = np.std(pnl_arr[:])

    # Calculate Sharpe ratio
    sharpe_ratio = (252)**0.5*average_return / volatility
    df_ret = pd.DataFrame({
    'TR': [np.round(np.sum(pnl_arr),2)],
    'MR': [average_return],
    'Risk': [volatility],
    'SR': [sharpe_ratio],
    'MDD': [calculate_maximum_drawdown(pnl_arr)]
    }, index=[0])
    return df_ret.round(2)

def create_weights_df(w_list,date):
    df_weights = pd.DataFrame(w_list)
    df_weights['date'] = date
    df_weights = df_weights.set_index('date')
    df_weights.columns  = ['TTF','POW','EUA']
    return df_weights