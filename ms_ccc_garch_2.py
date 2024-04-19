import numpy as np
import scipy
import scipy.linalg.lapack as lapack
from numpy import array, diag, exp, log, prod, sqrt, std, zeros
from scipy.linalg import expm, norm
from scipy.optimize import minimize
from scipy.special import gamma, kn, kv


class mgarch:
    """
    Implements the Multivariate Markov Switching CCC GARCH model of Markus Haas & Ji-Chun Liu [1].
    Main logic and functions are taken from Code Ocean implementation for a bivariate case. 
    Main adjustmens include:
    - Addition of algorithm for PSD parametrization of correlation matrix proposed by Archakov, I. & Reinhard Hansen, P [2].
    - Maximum Likelihood estimation is rewritten in vectorized fashion, which improves the speed of path dependant 
    calculations of GARCH parameters and transition probabilities. 
    -Cholesky decomposition is applied to covariance matrix which improves estimation 
    since matrix inverse operation is replaced by simplier operations. 
    - Included estimation of model with Skewed-t distribution of innovations.
    - Functions are rewritten to support estimation for n-variate case.
    - Integrates an estimation with an AR term, including formulas for forecasting the covariance matrix 
    in cases of autocorrelated returns

    Attributes:
        regime_mean (bool): Indicates if means are regime variant.
        ret (array): The returns or time series data being modeled.
        k (int): Number of regimes.
        ar (bool): Indicates if autoregressive terms are included.
        regime_ar (bool): Indicates if autoregressive parameters are regime variant.
        init_params (array): Initial parameters for optimization.
        last_params (array): Parameters from the last step of optimization.
        param_start (int): Initial index for parameter vector, dependent on regime modeling.
    
    Parameters:
        ret (array): The returns or time series data.
        dist (str, optional): The distribution for errors. Defaults to 'norm'.
        regime_mean (bool, optional): Model separate means for different regimes. Defaults to False.
        k (int, optional): Number of regimes.
        ar (bool, optional): Include autoregressive terms. Defaults to False.
        regime_ar (bool, optional): Separate AR parameters for each regime. Defaults to False.
        init_params (array, optional): Initial parameter guess. Defaults to an empty array.

    Note: class is not fully implemented/tested. Some specifications may theoretically result in errors.
    """

    def __init__(self,ret,dist='norm', regime_mean=False,k =2, ar = False,regime_ar = False, init_params = np.empty(0)):
        self.regime_mean = regime_mean
        self.ret = ret 
        self.k = k 
        self.ar = ar
        self.regime_ar = regime_ar
        self.init_params = init_params
        self.last_params = None
        self.param_start = 2 if regime_mean else 1

        if ar and regime_ar:
            self.ar_term  = 2 
        elif ar:
            self.ar_term  = 1 
        else:
            self.ar_term  = 0

        if dist in ['norm','t','skewed_t']:
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm','t','skewed_t'.")
    
    def cov_matrix_inverse_matrix(self,rho_vec,tol_value):
        C = []
        try:
            n = 0.5*(1+np.sqrt(1+8*len(rho_vec)))
            if not all([rho_vec.ndim == 1,n.is_integer(),1e-14 <= tol_value <=  1e-4 ]):
                raise ValueError
            n = int(n)
            A = np.zeros((n,n))
            A[np.triu_indices(n,1)] = rho_vec
            A = A + A.T
            diag_vec = np.diag(A)
            diag_ind = np.diag_indices_from(A)
            dist = np.sqrt(n)
            while dist > np.sqrt(n) * tol_value:
                diag_delta = np.log(np.diag(expm(A)))
                diag_vec = diag_vec - diag_delta
                A[diag_ind] = diag_vec
                dist = norm(diag_delta)
            C = expm(A)
            np.fill_diagonal(C,1)
        except:
            print('Error input ')
        return C
    
    def cov_to_corr(self,cov_matrix):
        n = cov_matrix.shape[0]
        corr_matrix = np.zeros_like(cov_matrix)
        
        for i in range(n):
            for j in range(n):
                corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                
        return corr_matrix

    
    def get_init_params(self):
        T, d = self.ret.shape
        off_diag = int(d * (d - 1) /2 )
        num_params = self.param_start*d + (self.ar_term * d ) + 2*d*3 + d + off_diag*2

        # add skew parameter vector for skewed_t
        if self.dist == 'skewed_t':
            num_params = num_params + d

        if (self.init_params== None).all() == False:
            num_plus = 2 if self.dist == 'norm' else 3 
            if self.init_params.shape[0] != num_params + num_plus:
                raise ValueError("Initial parameters vector doesnt match model scpecified")
            return self.init_params
        else:
            initial_ms = np.random.random(num_params)
            initial_ms = np.append(initial_ms,[0.95,0.95,8])
            initial_ms_map = np.zeros(len(initial_ms))

            if self.ar == True:
                if self.regime_ar==True:
                    initial_ms_map[(self.param_start)*d:(self.param_start+2)*d]  = np.log((1+initial_ms[(self.param_start)*d:(self.param_start+2)*d])/(1-initial_ms[(self.param_start)*d:(self.param_start+2)*d]))
                    initial_ms_map[(self.param_start+2)*d:(self.param_start+6)*d] = -np.log(initial_ms[(self.param_start+2)*d:(self.param_start+6)*d] )
                    initial_ms_map[(self.param_start+6)*d:(self.param_start+8)*d] = np.log(initial_ms[(self.param_start+6)*d:(self.param_start+8)*d]/(1-initial_ms[(self.param_start+6)*d:(self.param_start+8)*d]))
                    initial_ms_map[(self.param_start+8)*d:(self.param_start+9)*d]  = np.log((1+initial_ms[(self.param_start+8)*d:(self.param_start+9)*d])/(1-initial_ms[(self.param_start+8)*d:(self.param_start+9)*d]))
                    initial_ms_map[(self.param_start+9)*d:(self.param_start+9)*d + 2* off_diag]  = -log(4/(initial_ms[(self.param_start+9)*d:(self.param_start+9)*d + 2* off_diag]  + 2) - 1)   
                    if self.dist == 'skewed_t':
                        initial_ms_map[(self.param_start+9)*d + 2* off_diag:-3] = -log(4/(initial_ms[(self.param_start+9)*d + 2* off_diag:-3] + 2) - 1)   
                    initial_ms_map[-3:-1] = np.log(initial_ms[-3:-1]/(1-initial_ms[-3:-1]))
                    initial_ms_map[-1] = np.log(initial_ms[-1]- 2)
                    if self.dist == 'norm':
                        initial_ms_map = initial_ms_map[:-1]
                else:
                    initial_ms_map[(self.param_start)*d:(self.param_start+1)*d]  = np.log((1+initial_ms[(self.param_start)*d:(self.param_start+1)*d])/(1-initial_ms[(self.param_start)*d:(self.param_start+1)*d]))
                    initial_ms_map[(self.param_start+1)*d:(self.param_start+5)*d] = -np.log(initial_ms[(self.param_start+1)*d:(self.param_start+5)*d])
                    initial_ms_map[(self.param_start+5)*d:(self.param_start+7)*d] = np.log(initial_ms[(self.param_start+5)*d:(self.param_start+7)*d]/(1-initial_ms[(self.param_start+5)*d:(self.param_start+7)*d]))
                    initial_ms_map[(self.param_start+7)*d:(self.param_start+8)*d]  = np.log((1+initial_ms[(self.param_start+7)*d:(self.param_start+8)*d])/(1-initial_ms[(self.param_start+7)*d:(self.param_start+8)*d]))
                    initial_ms_map[(self.param_start+8)*d:(self.param_start+8)*d+2*off_diag]  = -log(4/(initial_ms[(self.param_start+8)*d:(self.param_start+8)*d+2*off_diag]  + 2) - 1)  
                    if self.dist == 'skewed_t':
                        initial_ms_map[(self.param_start+8)*d + 2* off_diag:-3] = -log(4/(initial_ms[(self.param_start+8)*d + 2* off_diag:-3] + 2) - 1)   
                    initial_ms_map[-3:-1] = np.log(initial_ms[-3:-1]/(1-initial_ms[-3:-1]))
                    initial_ms_map[-1] = np.log(initial_ms[-1]- 2)
                    if self.dist == 'norm':
                        initial_ms_map = initial_ms_map[:-1]
            else:
                initial_ms_map[self.param_start*d:(self.param_start+4)*d] = -np.log(initial_ms[self.param_start*d:(self.param_start+4)*d] )
                initial_ms_map[(self.param_start+4)*d:(self.param_start+6)*d] = np.log(initial_ms[(self.param_start+4)*d:(self.param_start+6)*d]/(1-initial_ms[(self.param_start+4)*d:(self.param_start+6)*d]))
                initial_ms_map[(self.param_start+6)*d:(self.param_start+7)*d]  = np.log((1+initial_ms[(self.param_start+6)*d:(self.param_start+7)*d])/(1-initial_ms[(self.param_start+6)*d:(self.param_start+7)*d]))
                initial_ms_map[(self.param_start+7)*d:(self.param_start+7)*d+2*off_diag]  = -log(4/(initial_ms[(self.param_start+7)*d:(self.param_start+7)*d+2*off_diag]  + 2) - 1) 
                if self.dist == 'skewed_t':
                    initial_ms_map[(self.param_start+7)*d + 2* off_diag:-3] = -log(4/(initial_ms[(self.param_start+7)*d + 2* off_diag:-3] + 2) - 1)   
                initial_ms_map[-3:-1] = np.log(initial_ms[-3:-1]/(1-initial_ms[-3:-1]))
                initial_ms_map[-1] = np.log(initial_ms[-1]- 2)
                if self.dist == 'norm':
                    initial_ms_map = initial_ms_map[:-1]
                    
            return initial_ms_map
        
    def skewed_t_pdf(self,eps,sigma,nu,gam,d):
        inv_sigma = np.linalg.inv(sigma) 
        gam_sig = (gam.T @ inv_sigma @ gam)
        nu_plus_d = (nu+d)/2
        c = ((nu-2)**(nu/2)* gam_sig**(nu_plus_d))/ ((2*np.pi)**(d/2) * np.sqrt(np.linalg.det(sigma)) * gamma(nu/2)*2**(nu/2-1))
        Q = (eps).T @ inv_sigma @ (eps)
        nugam = np.sqrt((nu - 2 + Q)*gam_sig)
        fx = c * ((kv(nu_plus_d,nugam) * np.exp((eps).T @ inv_sigma @ gam)) / (nugam**(nu_plus_d)))
        return fx

    def calculate_eps(self, mu, phi=None):
        if phi is not None:
            eps = self.ret - mu - phi * (np.roll(self.ret, 1, axis=0) - mu)
            return eps[1:, :]
        else:
            return self.ret - mu
    
    def filter_returns(self,theta):
        '''
        Filter returns mean and autocorrelation
        '''
        T, d = self.ret.shape
        if self.ar:
            if self.regime_ar:
                phi1 = -1 + 2 / (1 + np.exp(-theta[self.param_start * d:(self.param_start + 1) * d]))
                phi2 = -1 + 2 / (1 + np.exp(-theta[(self.param_start + 1) * d:(self.param_start + 2) * d]))
                T = T-1
            else:
                phi1 = -1 + 2 / (1 + np.exp(-theta[self.param_start * d:(self.param_start + 1) * d]))
                phi2 = phi1
                T = T-1
        else:
            phi1 = phi2 = None

        if self.regime_mean == True:
            mu,mu2 = theta[0:d],theta[d:2*d]
            eps1 = self.calculate_eps( mu, phi1)
            eps2 = self.calculate_eps( mu2, phi2)
        else:
            mu = theta[0:d]
            eps1 = self.calculate_eps( mu, phi1)
            eps2 = self.calculate_eps( mu, phi2)

        return eps1,eps2,phi1,phi2,T,d

    def get_garch_params(self,theta,d):
        '''
        Inverse map garch parameters for MLE
        '''
        w1,w2 = exp(-theta[(self.param_start+self.ar_term )*d:(self.param_start+1+self.ar_term )*d]),exp(-theta[(self.param_start+1+self.ar_term )*d:(self.param_start+2+self.ar_term )*d])

        A1,A2 = diag(exp(-theta[(self.param_start+2+self.ar_term )*d:(self.param_start+3+self.ar_term )*d])),diag(exp(-theta[(self.param_start+3+self.ar_term )*d:(self.param_start+4+self.ar_term )*d]))

        B1,B2 = diag(1/(1+exp(-theta[(self.param_start+4+self.ar_term )*d:(self.param_start+5+self.ar_term )*d]))),diag(1/(1+exp(-theta[(self.param_start+5+self.ar_term )*d:(self.param_start+6+self.ar_term )*d])))

        G = diag(-1+2/(1+np.exp(-theta[(self.param_start+6+self.ar_term )*d:(self.param_start+7+self.ar_term )*d])))
        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(self.param_start+7+self.ar_term )*d:(self.param_start+7+self.ar_term )*d+off_diag])),-2 + 4/(1+exp(-theta[(self.param_start+7+self.ar_term )*d+off_diag:(self.param_start+7+self.ar_term )*d+2*off_diag]))]

        # rho = [theta[(self.param_start+7)*d:(self.param_start+7)*d+off_diag],theta[(self.param_start+7)*d+off_diag:(self.param_start+7)*d+2*off_diag]]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]

        if self.dist == 'norm':
            p,q = 1/(1+exp(-theta[-2])),1/(1+exp(-theta[-1]))
        else:
            p,q = 1/(1+exp(-theta[-3])),1/(1+exp(-theta[-2]))

        if self.dist == 'skewed_t':
            gam = -2 + 4/(1+exp(-theta[-6:-3]))
            return w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam
        else:
            return w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q

    
    def loglik_stu(self,theta):
        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''

        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf
        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        const = scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        LL = np.zeros((2,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L2_mahalanobis_squared = (1 + np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        L1_density =  L1_mahalanobis_squared/L1_diag
        L2_density =  L2_mahalanobis_squared/L2_diag

        for t in range(1,T):
            LL[0,0] = ksi[0]*L1_density[t]
            LL[1,0] = ksi[1]*L2_density[t]
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] = np.log(np.sum(LL)) 

        loglik = np.sum(loglik[1:T]+ const)
            
        L = -loglik/T
        self.loglik = loglik
        
        return L 

    def loglik_skewed_t(self,theta):    
        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)

        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf
        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t

        LL = np.zeros((2,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        const = (nu-2)**(nu/2) / ((2*np.pi)**(d/2) * gamma(nu/2)*2**(nu/2-1))
        nu_plus_d = (nu+d)/2

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_inv_gam = np.linalg.solve(L1[:], np.tile(gam.T[:,None], (1,T))[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_gam = np.linalg.solve(L2[:], np.tile(gam.T[:,None], (1,T))[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)
        L2_mahalanobis_squared = np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)

        L1_gam_squared = np.sum(L1_inv_gam * L1_inv_gam, axis=1)
        L2_gam_squared = np.sum(L2_inv_gam * L2_inv_gam, axis=1)

        nugam1 = np.sqrt((nu - 2 + L1_mahalanobis_squared)*L1_gam_squared) + 1e-8
        nugam2  = np.sqrt((nu - 2 + L2_mahalanobis_squared)*L2_gam_squared)+ 1e-8

        nugam1_pow_nu = np.power(nugam1,nu_plus_d)
        nugam2_pow_nu = np.power(nugam2,nu_plus_d)

        L1_gam_squared_nu = np.power(np.sum(L1_inv_gam * L1_inv_gam, axis=1),(nu_plus_d))
        L2_gam_squared_nu = np.power(np.sum(L2_inv_gam * L2_inv_gam, axis=1),(nu_plus_d))

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        exp_L1_mahalanobis_gamma_div_nugam1_pow_nu = np.exp(np.sum(L1_inv_y_minus_mu * L1_inv_gam, axis=1)) / (nugam1_pow_nu)
        exp_L2_mahalanobis_gamma_div_nugam2_pow_nu = np.exp(np.sum(L2_inv_y_minus_mu * L2_inv_gam, axis=1)) / (nugam2_pow_nu)

        L1_gam_squared_nu_div_L1_diag = L1_gam_squared_nu / L1_diag
        L2_gam_squared_nu_div_L2_diag = L2_gam_squared_nu / L2_diag

        L1_bessel = kv(nu_plus_d,nugam1)
        L2_bessel = kv(nu_plus_d,nugam2)

        L1_density = L1_gam_squared_nu_div_L1_diag * L1_bessel * exp_L1_mahalanobis_gamma_div_nugam1_pow_nu
        L2_density = L2_gam_squared_nu_div_L2_diag * L2_bessel * exp_L2_mahalanobis_gamma_div_nugam2_pow_nu

        for t in range(1,T):
            LL[0,0] = ksi[0] * L1_density[t]
            LL[1,0] = ksi[1] * L2_density[t]
            xsi = LL/np.sum(LL)
            ksi= np.dot(P,xsi).ravel()
            loglik[t,0] = np.log(np.sum(LL)) 

        loglik = np.sum(loglik[1:T]+ np.log(const))
        L = -loglik/T
        return L 

    
    def loglik_norm(self,theta):
        '''
        Function estimating Log Likelihood of markov switching 2 regimes model with norm multivariate innovations
        '''
        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf

        # volatility
        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        
        LL = np.zeros((2,1))
        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        const = np.power((2.0 * np.pi), d / 2.0)
        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = exp(-0.5 *np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1))
        L2_mahalanobis_squared = exp(-0.5 *np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1))

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        L1_density =  1.0 / (const* L1_diag) * L1_mahalanobis_squared
        L2_density =  1.0 / (const* L2_diag) * L2_mahalanobis_squared


        for t in range(1,T):
            LL[0,0] = ksi[0]* L1_density[t]
            LL[1,0] = ksi[1]* L2_density[t]
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL))
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        L = -loglik/T
        return L 
    

    def smooth_prob_stu(self,theta):

        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)

        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))

        ksi=np.zeros([self.k ,T+1])
        
        xsi=np.zeros([self.k ,T])
        ksi[:,1] = pinf.T
        
        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        abs_eps1 = abs(eps1_t)
        abs_eps2 = abs(eps2_t)
        A1_dot_abs_eps1 = A1@abs_eps1
        A2_dot_abs_eps2 = A2@abs_eps2
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)

        condcorr = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)

        LL = np.zeros((2,1))
        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
            L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
            L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
            LL[0,0] = ksi[0,t]*(1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1[t]))
            LL[1,0] = ksi[1,t]*(1+L2_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L2[t]))
            fporb =LL/sum(LL)
            xsi[0,t] = fporb[0]
            xsi[1,t] = fporb[1]
            ksi[:,t+1] =np.dot(P,xsi[:,t])
            loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL)) + const
            sigmabar = ksi[0,t] *sigma1[t] + ksi[1,t] *sigma2[t]
            condcorr[t] = self.cov_to_corr(sigmabar)

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])

        chi = np.zeros([self.k ,T])
        chi[:,T-1] = xsi[:,T-1]

        for t in range(1,T):
            chi[:,T-t-1]=xsi[:,T-t-1]*(P.T@(chi[:,T-t]/ksi[:,T-t]))
            

        mu_list = [theta[0:d],theta[d:2*d]] if self.regime_mean else [theta[0:d]]

        if self.ar == True:
            if self.regime_ar == True:
                ar_list = [phi1,phi2]
            else:
                ar_list = [phi1]
        else:
            ar_list = []
        
        est = mu_list + ar_list + [w1,w2, np.diag(A1), np.diag(A2), np.diag(B1), np.diag(B2), np.diag(G),R1,R2,p,q,nu]

        return xsi,chi,condcorr,est,h1,h2,loglik

    def smooth_prob_norm(self,theta):

        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)

        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))

        # probabilities
        ksi=np.zeros([self.k ,T+1])
    
        xsi=np.zeros([self.k ,T])
        ksi[:,1] = pinf.T

        # volatility
        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        abs_eps1 = abs(eps1_t)
        abs_eps2 = abs(eps2_t)
        A1_dot_abs_eps1 = A1@abs_eps1
        A2_dot_abs_eps2 = A2@abs_eps2
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t

        LL = np.zeros((2,1))
        condcorr = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)

        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        const = np.power((2.0 * np.pi), d / 2.0)
        
        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
            L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
            L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
            LL[0,0] = ksi[0,t]* 1.0 / (const * prod(np.diag(L1[t]))) * exp(-0.5 * L1_mahalanobis_squared)
            LL[1,0] = ksi[1,t]* 1.0 / (const * prod(np.diag(L2[t]))) * exp(-0.5 * L2_mahalanobis_squared)
            fporb =LL/sum(LL)
            xsi[0,t] = fporb[0]
            xsi[1,t] = fporb[1]
            ksi[:,t+1] =np.dot(P,xsi[:,t])
            loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL))
            sigmabar = ksi[0,t] *sigma1[t] + ksi[1,t] *sigma2[t]
            condcorr[t] = self.cov_to_corr(sigmabar)

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        chi = np.zeros([self.k ,T])
        chi[:,T-1] = xsi[:,T-1]

        for t in range(1,T):
            chi[:,T-t-1]=xsi[:,T-t-1]*(P.T@(chi[:,T-t]/ksi[:,T-t]))

        mu_list = [theta[0:d],theta[d:2*d]] if self.regime_mean else [theta[0:d]]

        if self.ar == True:
            if self.regime_ar == True:
                ar_list = [phi1,phi2]
            else:
                ar_list = [phi1]
        else:
            ar_list = []
        
        est = mu_list + ar_list + [w1,w2, np.diag(A1), np.diag(A2), np.diag(B1), np.diag(B2), np.diag(G),R1,R2,p,q]


        return xsi,chi,condcorr,est,h1,h2,loglik
    

    def smooth_prob_log_skewed_t(self,theta):

        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)

        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))

        ksi=np.zeros([self.k ,T+1])
        
        xsi=np.zeros([self.k ,T])
        ksi[:,1] = pinf.T
        
        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        abs_eps1 = abs(eps1_t)
        abs_eps2 = abs(eps2_t)
        A1_dot_abs_eps1 = A1@abs_eps1
        A2_dot_abs_eps2 = A2@abs_eps2
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        condcorr = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)

        LL = np.zeros((2,1))
        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2

        for t in range(1,T):
            LL[0,0] = ksi[0,t]*self.skewed_t_pdf(eps1_t[:,t],sigma1[t],nu,gam,d)
            LL[1,0] = ksi[1,t]*self.skewed_t_pdf(eps2_t[:,t],sigma2[t],nu,gam,d)
            fporb =LL/sum(LL)
            xsi[0,t] = fporb[0]
            xsi[1,t] = fporb[1]
            ksi[:,t+1] =np.dot(P,xsi[:,t])
            loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL))
            sigmabar = ksi[0,t] *sigma1[t] + ksi[1,t] *sigma2[t]
            condcorr[t] = self.cov_to_corr(sigmabar)

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])

        chi = np.zeros([self.k ,T])
        chi[:,T-1] = xsi[:,T-1]

        for t in range(1,T):
            chi[:,T-t-1]=xsi[:,T-t-1]*(P.T@(chi[:,T-t]/ksi[:,T-t]))
            
        mu_list = [theta[0:d],theta[d:2*d]] if self.regime_mean else [theta[0:d]]

        if self.ar == True:
            if self.regime_ar == True:
                ar_list = [phi1,phi2]
            else:
                ar_list = [phi1]
        else:
            ar_list = []
        
        est = mu_list + ar_list + [w1,w2, np.diag(A1), np.diag(A2), np.diag(B1), np.diag(B2), np.diag(G),gam,R1,R2,p,q,nu]

        return xsi,chi,condcorr,est,h1,h2,loglik


    def get_params(self,theta):
        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)


        if  self.dist == 'skewed_t':
            w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam = self.get_garch_params(theta,d)
        else:
            w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)
        
        if self.regime_mean == True:
            mu,mu2 = theta[0:d],theta[d:2*d]
            mus = [mu,mu2]

        else:
            mus = theta[0:d]

        if self.dist == 't':  
            p,q = 1/(1+exp(-theta[-3])),1/(1+exp(-theta[-2]))
            nu = exp(theta[-1])+2
            return  np.concatenate((mus.ravel(),phi1,w1,w2,diag(A1),diag(A2),diag(B1),diag(B2),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],R2[np.triu_indices(R2.shape[0], k=1)],np.array([p,q,nu]))).ravel()
        elif self.dist == 'skewed_t':
            p,q = 1/(1+exp(-theta[-3])),1/(1+exp(-theta[-2]))
            nu = exp(theta[-1])+2
            return  np.concatenate((mus.ravel(),phi1,w1,w2,diag(A1),diag(A2),diag(B1),diag(B2),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],R2[np.triu_indices(R2.shape[0], k=1)],gam,np.array([p,q,nu]))).ravel()
        else:
            p,q = 1/(1+exp(-theta[-2])),1/(1+exp(-theta[-1]))
            return  np.concatenate((mus.ravel(),phi1,w1,w2,diag(A1),diag(A2),diag(B1),diag(B2),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],R2[np.triu_indices(R2.shape[0], k=1)],np.array([p,q]))).ravel()
        
    def params_for_fcst_ar(self,theta):
        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''
        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params(theta,d)


        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf


        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T+1]),zeros([d,T+1])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        abs_eps1 = abs(eps1_t)
        abs_eps2 = abs(eps2_t)
        A1_dot_abs_eps1 = A1@abs_eps1
        A2_dot_abs_eps2 = A2@abs_eps2
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        LL = np.zeros((2,1))

        for t in range(1,T+1):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        
        if self.dist == 't':
            const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
            for t in range(1,T):
                L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
                L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
                L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
                L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu

                LL[0,0] = ksi[0]*(1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1[t]))
                LL[1,0] = ksi[1]*(1+L2_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L2[t]))
                xsi = LL/np.sum(LL)
                ksi=np.dot(P,xsi)
                loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL)) + const
                
            loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
            pinit = xsi

            return w1.reshape(-1, 1) ,w2.reshape(-1, 1) ,phi1,phi2,A1,A2,B1,B2,G,p,q,nu,P,R1,R2,h1[:,-1],h2[:,-1],pinit
        else:
            const = np.power((2.0 * np.pi), d / 2.0)
            for t in range(1,T):
                L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
                L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
                L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
                L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
                LL[0,0] = ksi[0]* 1.0 / (const* prod(np.diag(L1[t]))) * exp(-0.5 * L1_mahalanobis_squared)
                LL[1,0] = ksi[1]* 1.0 / (const* prod(np.diag(L2[t]))) * exp(-0.5 * L2_mahalanobis_squared)
                xsi = LL/np.sum(LL)
                ksi=np.dot(P,xsi)
                loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL))
                
            loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
            pinit = xsi
            return w1.reshape(-1, 1) ,w2.reshape(-1, 1) ,phi1,phi2,A1,A2,B1,B2,G,p,q,P,R1,R2,h1[:,-1],h2[:,-1],pinit

    
    def params_for_fcst_ar_skewed_t(self,theta):
        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''
        eps1,eps2,phi1,phi2,T,d = self.filter_returns(theta)

        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam = self.get_garch_params(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf
        nu = exp(theta[-1])+2

        h1,h2 = zeros([d,T+1]),zeros([d,T+1])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        abs_eps1 = abs(eps1_t)
        abs_eps2 = abs(eps2_t)
        A1_dot_abs_eps1 = A1@abs_eps1
        A2_dot_abs_eps2 = A2@abs_eps2
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        LL = np.zeros((2,1))

        for t in range(1,T+1):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        const = (nu-2)**(nu/2) / ((2*np.pi)**(d/2) * gamma(nu/2)*2**(nu/2-1))
        nu_plus_d = (nu+d)/2


        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
            L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
            L1_inv_gam, _  = lapack.dtrtrs(L1[t], gam, lower=True)
            L2_inv_gam, _  = lapack.dtrtrs(L2[t], gam, lower=True)

            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
            L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
            L1_gam_squared = L1_inv_gam.T @ L1_inv_gam
            L2_gam_squared = L2_inv_gam.T @ L2_inv_gam
            
            nugam1  = np.sqrt((nu - 2 + L1_mahalanobis_squared)*L1_gam_squared)
            nugam2  = np.sqrt((nu - 2 + L2_mahalanobis_squared)*L2_gam_squared)

            LL[0,0] = ksi[0] * ((L1_gam_squared**(nu_plus_d)) / (prod(np.diag(L1[t])))*(kv(nu_plus_d,nugam1) * np.exp(L1_inv_y_minus_mu @ L1_inv_gam)) / (nugam1 **(nu_plus_d)))
            LL[1,0] = ksi[1] * ((L2_gam_squared**(nu_plus_d)) / (prod(np.diag(L2[t])))*(kv(nu_plus_d,nugam2) * np.exp(L2_inv_y_minus_mu @ L2_inv_gam)) / (nugam2 **(nu_plus_d)))
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] = np.log(np.dot(np.ones((1,2)),LL)) + np.log(const)
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        pinit = ksi
        return w1,w2,phi1,phi2,A1,A2,B1,B2,G,p,q,nu,P,R1,R2,h1[:,-1],h2[:,-1],pinit,gam
    
    def fill_correlation_matrix(self,correlations):
        n = len(correlations)
        corr_matrix = np.eye(n)  # Initialize a diagonal matrix with ones
        k = 0  # Index for traversing the correlations vector
        
        for i in range(n-1):
            for j in range(i+1, n):
                corr_matrix[i, j] = correlations[k]
                corr_matrix[j, i] = correlations[k]
                k += 1
        
        return corr_matrix
    
    def filter_returns_err(self,theta):
        '''
        Filter returns mean and autocorrelation
        '''
        T, d = self.ret.shape
        if self.ar:
            if self.regime_ar:
                phi1 = theta[self.param_start * d:(self.param_start + 1) * d]
                phi2 = theta[(self.param_start + 1) * d:(self.param_start + 2) * d]
                T = T-1
            else:
                phi1 = theta[self.param_start * d:(self.param_start + 1) * d]
                phi2 = phi1
                T = T-1
        else:
            phi1 = phi2 = None

        if self.regime_mean == True:
            mu,mu2 = theta[0:d],theta[d:2*d]
            eps1 = self.calculate_eps( mu, phi1)
            eps2 = self.calculate_eps( mu2, phi2)
        else:
            mu = theta[0:d]
            eps1 = self.calculate_eps( mu, phi1)
            eps2 = self.calculate_eps( mu, phi2)

        return eps1,eps2,phi1,phi2,T,d
    
    
    def get_garch_params_err(self,theta,d):
        '''
        Inverse map garch untransfomed parameters for std err calculation. 
        '''
        w1,w2 = theta[(self.param_start+self.ar_term )*d:(self.param_start+1+self.ar_term )*d],theta[(self.param_start+1+self.ar_term )*d:(self.param_start+2+self.ar_term )*d]

        A1,A2 = diag(theta[(self.param_start+2+self.ar_term )*d:(self.param_start+3+self.ar_term )*d]),diag(theta[(self.param_start+3+self.ar_term )*d:(self.param_start+4+self.ar_term )*d])

        B1,B2 = diag(theta[(self.param_start+4+self.ar_term )*d:(self.param_start+5+self.ar_term )*d]),diag(theta[(self.param_start+5+self.ar_term )*d:(self.param_start+6+self.ar_term )*d])

        G = diag(theta[(self.param_start+6+self.ar_term )*d:(self.param_start+7+self.ar_term )*d])
        
        off_diag = int(d * (d - 1) /2 )
        rho = [theta[(self.param_start+7+self.ar_term )*d:(self.param_start+7+self.ar_term )*d+off_diag],theta[(self.param_start+7+self.ar_term )*d+off_diag:(self.param_start+7+self.ar_term )*d+2*off_diag]]

        # rho = [theta[(self.param_start+7)*d:(self.param_start+7)*d+off_diag],theta[(self.param_start+7)*d+off_diag:(self.param_start+7)*d+2*off_diag]]
        
        R1 = self.fill_correlation_matrix(rho[0])
        R2 = self.fill_correlation_matrix(rho[1])

        if self.dist == 'norm':
            p,q = theta[-2],theta[-1]
        else:
            p,q = theta[-3],theta[-2]

        if self.dist == 'skewed_t':
            gam = theta[-6:-3]
            return w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam
        else:
            return w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q

    

    def stu_err(self,theta):
        '''
        Function estimating markov switching student t loglik for computing standard errors
        '''

        eps1,eps2,phi1,phi2,T,d = self.filter_returns_err(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params_err(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf
        nu = theta[-1]

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        const = scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        LL = np.zeros((2,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L2_mahalanobis_squared = (1 + np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        L1_density =  L1_mahalanobis_squared/L1_diag
        L2_density =  L2_mahalanobis_squared/L2_diag

        for t in range(1,T):
            LL[0,0] = ksi[0]*L1_density[t]
            LL[1,0] = ksi[1]*L2_density[t]
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] = np.log(np.sum(LL)) 

        loglik = np.sum(loglik[1:T]+ const)

        
        return loglik 
    
    def norm_err(self,theta):

        eps1,eps2,phi1,phi2,T,d = self.filter_returns_err(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q = self.get_garch_params_err(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf

        # volatility
        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        
        LL = np.zeros((2,1))
        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        const = np.power((2.0 * np.pi), d / 2.0)
        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = exp(-0.5 *np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1))
        L2_mahalanobis_squared = exp(-0.5 *np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1))

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        L1_density =  1.0 / (const* L1_diag) * L1_mahalanobis_squared
        L2_density =  1.0 / (const* L2_diag) * L2_mahalanobis_squared


        for t in range(1,T):
            LL[0,0] = ksi[0]* L1_density[t]
            LL[1,0] = ksi[1]* L2_density[t]
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] =np.log(np.dot(np.ones((1,2)),LL))
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])

        return loglik
    
    def skewed_t_err(self,theta):    

        eps1,eps2,phi1,phi2,T,d = self.filter_returns_err(theta)
        w1,w2,A1,A2,B1,B2,G,off_diag,rho,R1,R2,p,q,gam = self.get_garch_params_err(theta,d)

        P = array([[p, 1 - q], [1 - p, q]])
        pinf = array(((1-q)/(2-p-q),1 - (1-q)/(2-p-q)))
        ksi = pinf
        nu = theta[-1]

        h1,h2 = zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0] = std(eps1,0),std(eps2,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t

        LL = np.zeros((2,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)

        const = (nu-2)**(nu/2) / ((2*np.pi)**(d/2) * gamma(nu/2)*2**(nu/2-1))
        nu_plus_d = (nu+d)/2

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)

        L1_inv_gam = np.linalg.solve(L1[:], np.tile(gam.T[:,None], (1,T))[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_gam = np.linalg.solve(L2[:], np.tile(gam.T[:,None], (1,T))[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)
        L2_mahalanobis_squared = np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)

        L1_gam_squared = np.sum(L1_inv_gam * L1_inv_gam, axis=1)
        L2_gam_squared = np.sum(L2_inv_gam * L2_inv_gam, axis=1)

        nugam1 = np.sqrt((nu - 2 + L1_mahalanobis_squared)*L1_gam_squared) + 1e-8
        nugam2  = np.sqrt((nu - 2 + L2_mahalanobis_squared)*L2_gam_squared)+ 1e-8

        nugam1_pow_nu = np.power(nugam1,nu_plus_d)
        nugam2_pow_nu = np.power(nugam2,nu_plus_d)

        L1_gam_squared_nu = np.power(np.sum(L1_inv_gam * L1_inv_gam, axis=1),(nu_plus_d))
        L2_gam_squared_nu = np.power(np.sum(L2_inv_gam * L2_inv_gam, axis=1),(nu_plus_d))

        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)

        exp_L1_mahalanobis_gamma_div_nugam1_pow_nu = np.exp(np.sum(L1_inv_y_minus_mu * L1_inv_gam, axis=1)) / (nugam1_pow_nu)
        exp_L2_mahalanobis_gamma_div_nugam2_pow_nu = np.exp(np.sum(L2_inv_y_minus_mu * L2_inv_gam, axis=1)) / (nugam2_pow_nu)

        L1_gam_squared_nu_div_L1_diag = L1_gam_squared_nu / L1_diag
        L2_gam_squared_nu_div_L2_diag = L2_gam_squared_nu / L2_diag

        L1_bessel = kv(nu_plus_d,nugam1)
        L2_bessel = kv(nu_plus_d,nugam2)

        L1_density = L1_gam_squared_nu_div_L1_diag * L1_bessel * exp_L1_mahalanobis_gamma_div_nugam1_pow_nu
        L2_density = L2_gam_squared_nu_div_L2_diag * L2_bessel * exp_L2_mahalanobis_gamma_div_nugam2_pow_nu

        for t in range(1,T):
            LL[0,0] = ksi[0] * L1_density[t]
            LL[1,0] = ksi[1] * L2_density[t]
            xsi = LL/np.sum(LL)
            ksi= np.dot(P,xsi).ravel()
            loglik[t,0] = np.log(np.sum(LL)) 

        loglik = np.sum(loglik[1:T]+ np.log(const))

        return loglik
    
    def fit(self):
        def callback_func(params):
            self.last_params = params  # Update the last_params attribute with the current parameters

        try:
            # Modify the optimization call to include a callback function that updates the last known parameters
            if self.dist == 't':
                res = minimize(self.loglik_stu, self.get_init_params(), callback=callback_func)
            elif self.dist == 'norm':
                res = minimize(self.loglik_norm, self.get_init_params(), callback=callback_func)
            elif self.dist == 'skewed_t':
                res = minimize(self.loglik_skewed_t, self.get_init_params(), callback=callback_func)
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            res = None
            if self.last_params is not None:
                print("The last set of parameters before failure is available in last_params attribute.")
                # Here, you can decide what to do with the last known parameters
                # For example, you might want to return them or use them in some way
        return res
    
    def get_smoothed_prob(self,params):
        if self.dist == 't':
            xsi,chi,condcorr,est,h1,h2,loglik = self.smooth_prob_stu(params)

        elif self.dist == 'norm':
            xsi,chi,condcorr,est,h1,h2,loglik = self.smooth_prob_norm(params)

        elif self.dist == 'skewed_t':
            xsi,chi,condcorr,est,h1,h2,loglik = self.smooth_prob_log_skewed_t(params)

        return xsi,chi,condcorr,est,h1,h2,loglik
    
    def forecastst(self, w1, w2, A1, A2, B1, B2, P, R1, R2, G1, G2, sigma1init, sigma2init, pinit,nu, D):

        pinit = np.array(pinit).reshape(-1, 1)
        Xinit = np.array([sigma1init, sigma2init]).ravel().reshape(-1, 1)
        XXinit = (Xinit @ Xinit.T).reshape(-1,1)
        Yinit = np.concatenate((Xinit, XXinit))

        kappa1 = np.sqrt(nu-2)/np.sqrt(np.pi)*exp(scipy.special.loggamma((nu-1)/2)-scipy.special.loggamma(nu/2))
        M = len(w1)
        R1.shape[0]
        k = P.shape[0]

        p = P[0, 0]
        q = P[1, 1]


        omega = np.vstack((w1, w2))
        A = np.vstack((A1, A2))
        Atilde = np.vstack((A1 * G1, A2 * G2))
        B = np.block([[B1, np.zeros((M, M))], [np.zeros((M, M)), B2]])

        Gam = np.zeros((M*M, M*M, 2))
        Gam[:, :, 0] = np.diag((2 / np.pi * (R1* np.arcsin(R1) + np.sqrt(1 - R1 ** 2))).reshape(1,-1)[0])
        Gam[:, :, 1] = np.diag((2 / np.pi * (R2* np.arcsin(R2) + np.sqrt(1 - R2 ** 2))).reshape(1,-1)[0])


        R = np.zeros((M, M, 2))
        G = np.zeros((M, M, 2))
        e = np.eye(2)

        R[:, :, 0] = R1
        R[:, :, 1] = R2
        G[:, :, 0] = G1
        G[:, :, 1] = G2

        R1tilde = np.vstack((np.hstack((R1,R1)), np.hstack((R1,R1))))
        R2tilde = np.vstack((np.hstack((R2,R2)), np.hstack((R2,R2))))
        Gam1tilde = np.diag(np.ravel(R1tilde))
        Gam2tilde = np.diag(np.ravel(R2tilde))


        Pgamtilde = np.vstack((np.hstack((p * Gam1tilde, (1 - q) * Gam1tilde)),np.hstack(((1 - p) * Gam2tilde, q * Gam2tilde))))

        C11 = np.zeros((M*k, M*k, k))
        C21 = np.zeros((k**2 * M**2, k * M, k))
        C22 = np.zeros((k**2 * M**2, k**2 * M**2, k))
        Ctilde = np.zeros((k**2 * M**2 + k*M, k**2 * M**2+ k*M, k))

        for j in range(k):
            C11[:, :, j] = kappa1 * (A @ np.kron(e[j, :], np.eye(M))) + B
            C21[:, :, j] = np.kron(omega, C11[:, :, j]) + np.kron(C11[:, :, j], omega)
            C22[:, :, j] = (np.kron(A, A) @ Gam[:, :, j] + np.kron(Atilde, Atilde) @ np.diag(np.ravel(R[:, :, j]))) @ np.kron(np.kron(e[j, :], np.eye(M)), np.kron(e[j, :], np.eye(M))) + \
            kappa1 * (np.kron(np.kron(e[j, :], A), B) + np.kron(B, np.kron(e[j, :], A))) + \
            np.kron(B, B)
            Ctilde[:, :, j] = np.block([[C11[:, :, j], np.zeros((k*M,k**2 * M**2))], [C21[:, :, j], C22[:, :, j]]])

        PCtilde = np.block([[p*Ctilde[:,:,0],(1-q)*Ctilde[:,:,0]],[(1-p)*Ctilde[:,:,1],q*Ctilde[:,:,1]]])
        omegatilde = np.vstack((omega,np.kron(omega,omega)))

        II = np.kron(np.eye(k), np.concatenate((np.zeros((k**2 * M**2, k * M)), np.eye(k**2 * M**2)), axis=1))
        Ytilde = np.zeros((Yinit.shape[0] * 2 , D))
        Ytilde[:, 0] = np.kron(pinit, Yinit).reshape(1,-1)

        sigmafore = np.zeros((M*M , D))

        condcorr = np.eye(M) * np.expand_dims(zeros([M,D]).T, axis=2)
        condcov = np.eye(M) * np.expand_dims(zeros([M,D]).T, axis=2)

        sigmafore[:, 0] = (np.kron(e[0, :], np.kron(np.kron(e[0, :], np.eye(M)), np.kron(e[0, :], np.eye(M)))) +
                        np.kron(e[1, :], np.kron(np.kron(e[1, :], np.eye(M)), np.kron(e[1, :], np.eye(M))))).dot(Pgamtilde).dot(II).dot(Ytilde[:, 0])
        C = np.reshape(sigmafore[:, 0], (M, M))
        condcorr[0] = self.cov_to_corr(C)
        condcov[0] = C
        prob = pinit

        for d in range(1, D):
            prob = np.dot(P, prob)
            Ytilde[:, d] = (np.kron(prob, omegatilde) + PCtilde @ Ytilde[:, d-1].reshape(-1,1)).flatten()
            temp = np.dot(np.dot(Pgamtilde, II), Ytilde[:, d])
            sigmafore[:, d] = (np.kron(e[0, :], np.kron(np.kron(e[0, :], np.eye(M)), np.kron(e[0, :], np.eye(M)))) +
                        np.kron(e[1, :], np.kron(np.kron(e[1, :], np.eye(M)), np.kron(e[1, :], np.eye(M))))).dot(temp)
            C = np.reshape(sigmafore[:, d], (M, M))
            condcorr[d] = self.cov_to_corr(C)
            condcov[d] = C
        
        return sigmafore, condcov, condcorr
    
    def forecasts_norm(self,w1, w2, A1, A2, B1, B2, P, R1, R2, G1, G2, sigma1init, sigma2init, pinit, D):

        pinit = np.array(pinit).reshape(-1, 1)
        Xinit = np.array([sigma1init, sigma2init]).ravel().reshape(-1, 1)
        XXinit = (Xinit @ Xinit.T).reshape(-1,1)
        Yinit = np.concatenate((Xinit, XXinit))

        kappa1 = np.sqrt(2/np.pi)
        M = len(w1)
        R1.shape[0]
        k = P.shape[0]

        p = P[0, 0]
        q = P[1, 1]


        omega = np.vstack((w1, w2))
        A = np.vstack((A1, A2))
        Atilde = np.vstack((A1 * G1, A2 * G2))
        B = np.block([[B1, np.zeros((M, M))], [np.zeros((M, M)), B2]])

        Gam = np.zeros((M*M, M*M, 2))
        Gam[:, :, 0] = np.diag((2 / np.pi * (R1* np.arcsin(R1) + np.sqrt(1 - R1 ** 2))).reshape(1,-1)[0])
        Gam[:, :, 1] = np.diag((2 / np.pi * (R2* np.arcsin(R2) + np.sqrt(1 - R2 ** 2))).reshape(1,-1)[0])


        R = np.zeros((M, M, 2))
        G = np.zeros((M, M, 2))
        e = np.eye(2)

        R[:, :, 0] = R1
        R[:, :, 1] = R2
        G[:, :, 0] = G1
        G[:, :, 1] = G2

        R1tilde = np.vstack((np.hstack((R1,R1)), np.hstack((R1,R1))))
        R2tilde = np.vstack((np.hstack((R2,R2)), np.hstack((R2,R2))))
        Gam1tilde = np.diag(np.ravel(R1tilde))
        Gam2tilde = np.diag(np.ravel(R2tilde))


        Pgamtilde = np.vstack((np.hstack((p * Gam1tilde, (1 - q) * Gam1tilde)),np.hstack(((1 - p) * Gam2tilde, q * Gam2tilde))))

        C11 = np.zeros((M*k, M*k, k))
        C21 = np.zeros((k**2 * M**2, k * M, k))
        C22 = np.zeros((k**2 * M**2, k**2 * M**2, k))
        Ctilde = np.zeros((k**2 * M**2 + k*M, k**2 * M**2+ k*M, k))

        for j in range(k):
            C11[:, :, j] = kappa1 * (A @ np.kron(e[j, :], np.eye(M))) + B
            C21[:, :, j] = np.kron(omega, C11[:, :, j]) + np.kron(C11[:, :, j], omega)
            C22[:, :, j] = (np.kron(A, A) @ Gam[:, :, j] + np.kron(Atilde, Atilde) @ np.diag(np.ravel(R[:, :, j]))) @ np.kron(np.kron(e[j, :], np.eye(M)), np.kron(e[j, :], np.eye(M))) + \
            kappa1 * (np.kron(np.kron(e[j, :], A), B) + np.kron(B, np.kron(e[j, :], A))) + \
            np.kron(B, B)
            Ctilde[:, :, j] = np.block([[C11[:, :, j], np.zeros((k*M,k**2 * M**2))], [C21[:, :, j], C22[:, :, j]]])

        PCtilde = np.block([[p*Ctilde[:,:,0],(1-q)*Ctilde[:,:,0]],[(1-p)*Ctilde[:,:,1],q*Ctilde[:,:,1]]])
        omegatilde = np.vstack((omega,np.kron(omega,omega)))

        II = np.kron(np.eye(k), np.concatenate((np.zeros((k**2 * M**2, k * M)), np.eye(k**2 * M**2)), axis=1))
        Ytilde = np.zeros((Yinit.shape[0] * 2 , D))
        Ytilde[:, 0] = np.kron(pinit, Yinit).reshape(1,-1)

        sigmafore = np.zeros((M*M , D))

        condcorr = np.eye(M) * np.expand_dims(zeros([M,D]).T, axis=2)
        condcov = np.eye(M) * np.expand_dims(zeros([M,D]).T, axis=2)

        sigmafore[:, 0] = (np.kron(e[0, :], np.kron(np.kron(e[0, :], np.eye(M)), np.kron(e[0, :], np.eye(M)))) +
                        np.kron(e[1, :], np.kron(np.kron(e[1, :], np.eye(M)), np.kron(e[1, :], np.eye(M))))).dot(Pgamtilde).dot(II).dot(Ytilde[:, 0])
        C = np.reshape(sigmafore[:, 0], (M, M))
        condcorr[0] = self.cov_to_corr(C)
        condcov[0] = C
        prob = pinit

        for d in range(1, D):
            prob = np.dot(P, prob)
            Ytilde[:, d] = (np.kron(prob, omegatilde) + PCtilde @ Ytilde[:, d-1].reshape(-1,1)).flatten()
            temp = np.dot(np.dot(Pgamtilde, II), Ytilde[:, d])
            sigmafore[:, d] = (np.kron(e[0, :], np.kron(np.kron(e[0, :], np.eye(M)), np.kron(e[0, :], np.eye(M)))) +
                        np.kron(e[1, :], np.kron(np.kron(e[1, :], np.eye(M)), np.kron(e[1, :], np.eye(M))))).dot(temp)
            C = np.reshape(sigmafore[:, d], (M, M))
            condcorr[d] = self.cov_to_corr(C)
            condcov[d] = C
        
        return sigmafore, condcov, condcorr

