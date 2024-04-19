from scipy.optimize import minimize
import numpy as np
import scipy
from scipy.special import gamma
from scipy.linalg import norm,expm
import scipy.linalg.lapack as lapack
from numpy import exp,diag,zeros,array,std,log,sqrt,prod
import numpy as np
from scipy.stats import multivariate_t

class dcc:
    def __init__(self,ret,dist='norm'):
        self.ret = ret 
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")

    def norm_dcc_mult(self,theta):
        T, d = self.ret.shape
        mu = theta[:d]
        eps = self.ret - mu

        T = eps.shape[0]

        # GARCH-Parameter
        w = np.exp(-theta[d:2*d])
        A = np.diag(np.exp(-theta[2*d:3*d]))
        B = np.diag(1 / (1 + np.exp(-theta[3*d:4*d])))
        G = diag(-1+2/(1+np.exp(-theta[4*d:5*d])))

        alpha = np.exp(theta[5*d]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        beta = np.exp(theta[5*d+1]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        h = np.zeros((d, T))
        h[:, 0] = np.std(eps, axis=0)
        loglik = np.zeros((T, 1))
        eps = eps.T
        epsst = np.zeros((d, T))
        epsst[:, 0] = np.diag(1 / h[:, 0]) @ eps[:, 0]

        
        for t in range(1, T):
            h[:, t] = w + A @ np.abs(eps[:, t-1]) - A @ G @ eps[:, t-1] + B @ h[:, t-1]
            epsst[:, t] = np.diag(1 / h[:, t]) @ eps[:, t]


        S = np.corrcoef(epsst)
        Q = S.copy()
        rho = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)
        for t in range(1, T):
            Q = (1 - alpha - beta) * S + alpha * np.outer(epsst[:, t-1], epsst[:, t-1]) + beta * Q
            R = diag(1/np.sqrt(diag(Q))) @ Q @ diag(1/np.sqrt(diag(Q)))
            sigma = diag(h[:,t]) @ R @ diag(h[:,t])
            L1 = np.linalg.cholesky(sigma)
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1, eps[:,t], lower=True)
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu

            loglik[t, 0] = np.log(1.0 / (np.power((2.0 * np.pi), d / 2.0) * prod(np.diag(L1))) * exp(-0.5 * L1_mahalanobis_squared))

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        L = -loglik/T
        self.loglik = loglik
        self.est = [mu,w,np.diag(A), np.diag(B), np.diag(G),alpha,beta]
        return L
    
    def stu_dcc_mult(self,theta):
        T, d = self.ret.shape
        mu = theta[:d]
        eps = self.ret  - mu

        T = eps.shape[0]

        # GARCH-Parameter
        w = np.exp(-theta[d:2*d])
        A = np.diag(np.exp(-theta[2*d:3*d]))
        B = np.diag(1 / (1 + np.exp(-theta[3*d:4*d])))
        G = diag(-1+2/(1+np.exp(-theta[4*d:5*d])))

        alpha = np.exp(theta[5*d]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        beta = np.exp(theta[5*d+1]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        nu = np.exp(theta[5*d+2]) + 2
        h = np.zeros((d, T))
        h[:, 0] = np.std(eps, axis=0)
        loglik = np.zeros((T, 1))
        eps = eps.T
        epsst = np.zeros((d, T))
        epsst[:, 0] = np.diag(1 / h[:, 0]) @ eps[:, 0]

        
        for t in range(1, T):
            h[:, t] = w + A @ np.abs(eps[:, t-1]) - A @ G @ eps[:, t-1] + B @ h[:, t-1]
            epsst[:, t] = np.diag(1 / h[:, t]) @ eps[:, t]


        S = np.corrcoef(epsst)
        Q = S.copy()
        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        for t in range(1, T):
            Q = (1 - alpha - beta) * S + alpha * np.outer(epsst[:, t-1], epsst[:, t-1]) + beta * Q
            R = diag(1/np.sqrt(diag(Q))) @ Q @ diag(1/np.sqrt(diag(Q)))
            sigma = diag(h[:,t]) @ R @ diag(h[:,t])
            L1 = np.linalg.cholesky(sigma)
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1, eps[:,t], lower=True)
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu

            loglik[t, 0] = np.log((1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1))) + const

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        L = -loglik/T
        self.loglik = loglik
        self.est = [mu,w,np.diag(A), np.diag(B), np.diag(G),alpha,beta,nu]
        return L
    
    def fit(self):
        if self.dist == 't':
            ddc_l = np.random.random(18)
            res = minimize(self.stu_dcc_mult, ddc_l)
        elif self.dist == 'norm':
            ddc_l = np.random.random(17)
            res = minimize(self.norm_dcc_mult, ddc_l)
        return res
    

    def tdccfore(self,theta):
        T, d =  self.ret.shape
        mu = theta[:d]
        eps =  self.ret - mu

        T = eps.shape[0]

        # GARCH-Parameter
        w = np.exp(-theta[d:2*d])
        A = np.diag(np.exp(-theta[2*d:3*d]))
        B = np.diag(1 / (1 + np.exp(-theta[3*d:4*d])))
        G = diag(-1+2/(1+np.exp(-theta[4*d:5*d])))

        alpha = np.exp(theta[5*d]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        beta = np.exp(theta[5*d+1]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        nu = np.exp(theta[5*d+2]) + 2
        h = np.std(eps, axis=0)
        eps = eps.T
        epsst = np.zeros((d, T))
        epsst[:, 0] = np.diag(1.0 / h) @ eps[:, 0]
        for t in range(1, T + 1):
            h =  w + A @ np.abs(eps[:, t-1]) - A @ G @ eps[:, t-1] + B @ h
            if t < T:
                epsst[:, t] = np.diag(1.0 / h) @ eps[:, t]
        S = np.corrcoef(epsst)
        Q = S
        for t in range(1, T + 1):
            Q = (1 - alpha - beta) * S + alpha * np.outer(epsst[:, t - 1], epsst[:, t - 1]) + beta * Q

        R = np.linalg.inv(np.eye(d) * np.sqrt(np.diag(Q))) @ Q @ np.linalg.inv(np.eye(d) * np.sqrt(np.diag(Q)))
    
        return Q, R, h, S, w, A, B, G, alpha, beta, nu
    
    def dccsim(self,Q, R, h, S, w, A, B, G, alpha, beta, nu, D):
            T, N = self.ret.shape

            sigma = np.zeros((N**2, D))
            for d in range(D):
                s = np.diag(h) @ R @ np.diag(h)
                sigma[:, d] = s.flatten()
                sqrt_nu = np.sqrt(nu - 2) / np.sqrt(nu)
                r = multivariate_t.rvs(loc=None, shape=R, df=nu, size=1) @ np.diag(h) * sqrt_nu
                epsst = r @ np.diag(1.0 / h)
                h = w + A @ np.abs(r.T) - A @ G @ r.T + B @ h
                Q = (1 - alpha - beta) * S + alpha * (epsst.T @ epsst) + beta * Q
                R = np.linalg.inv(np.eye(N) * np.sqrt(np.diag(Q))) @ Q @ np.linalg.inv(np.eye(N) * np.sqrt(np.diag(Q)))
            return sigma
    
    def tdccforecast(self,Q, R, h, S, w, A, B, G, alpha, beta, nu, D):

        T, N = self.ret.shape
        sigma = np.zeros((N**2, D))
        for k in range(1, 10001):  # Python ranges are non-inclusive of the end, hence 10001
            sigma_k = self.dccsim(Q, R, h, S, w, A, B, G, alpha, beta, nu, D)
            sigma = (sigma * (k - 1) + sigma_k) / k
        return sigma
    
    def stu_dcc_corr_mult(self,theta):
        T, d = self.ret.shape
        mu = theta[:d]
        eps = self.ret - mu

        T = eps.shape[0]

        # GARCH-Parameter
        w = np.exp(-theta[d:2*d])
        A = np.diag(np.exp(-theta[2*d:3*d]))
        B = np.diag(1 / (1 + np.exp(-theta[3*d:4*d])))
        G = diag(-1+2/(1+np.exp(-theta[4*d:5*d])))

        alpha = np.exp(theta[5*d]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        beta = np.exp(theta[5*d+1]) / (1 + np.exp(theta[5*d]) + np.exp(theta[5*d+1]))
        nu = np.exp(theta[5*d+2]) + 2
        h = np.zeros((d,T))
        h[:, 0] = np.std(eps, axis=0)
        loglik = np.zeros((T, 1))
        eps = eps.T
        epsst = np.zeros((d, T))
        epsst[:, 0] = np.diag(1 / h[:, 0]) @ eps[:, 0]

        
        for t in range(1, T):
            h[:, t] = w + A @ np.abs(eps[:, t-1]) - A @ G @ eps[:, t-1] + B @ h[:, t-1]
            epsst[:, t] = np.diag(1 / h[:, t]) @ eps[:, t]

        S = np.corrcoef(epsst)
        Q = S.copy()
        rho = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)
        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        for t in range(1, T):
            Q = (1 - alpha - beta) * S + alpha * np.outer(epsst[:, t-1], epsst[:, t-1]) + beta * Q
            R = diag(1/np.sqrt(diag(Q))) @ Q @ diag(1/np.sqrt(diag(Q)))
            rho[t] = R
            sigma = diag(h[:,t]) @ R @ diag(h[:,t])
            L1 = np.linalg.cholesky(sigma)
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1, eps[:,t], lower=True)
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu

            loglik[t, 0] = np.log((1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1))) + const

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        return rho