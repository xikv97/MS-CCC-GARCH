from scipy.special import hankel1, hankel2,gamma,kv,kn
from scipy.linalg import norm,expm
import numpy as np
from scipy.optimize import minimize
import scipy
import scipy.linalg.lapack as lapack
from numpy import exp,diag,zeros,array,std,log,sqrt,prod

class mgarch_single:
    """
    Implements the Multivariate CCC GARCH model of Markus Haas & Ji-Chun Liu [1].
    Please find description of main changes in ms_ccc_garch_2.py"""
    def __init__(self,ret,dist='norm',k = 1,init_params = np.empty(0)):
        self.ret = ret 
        self.k = k 
        self.init_params = init_params
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")

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
    
    def get_init_params(self):
        if (self.init_params== None).all() == False:
            return self.init_params
        else:
            param_start = 1

            T, d = self.ret.shape
            off_diag = int(d * (d - 1) /2 )

            initial_ms = np.random.random(param_start*d + d*3 + d + off_diag)
            initial_ms = np.append(initial_ms,[8])
            initial_ms_map = np.zeros(len(initial_ms))
            initial_ms_map[param_start*d:(param_start+2)*d] = -np.log(initial_ms[param_start*d:(param_start+2)*d] )
            initial_ms_map[(param_start+2)*d:(param_start+3)*d] = np.log(initial_ms[(param_start+2)*d:(param_start+3)*d]/(1-initial_ms[(param_start+2)*d:(param_start+3)*d]))
            initial_ms_map[(param_start+3)*d:(param_start+4)*d]  = np.log((1+initial_ms[(param_start+3)*d:(param_start+4)*d])/(1-initial_ms[(param_start+3)*d:(param_start+4)*d]))
            initial_ms_map[(param_start+4)*d:-1]  = -log(4/(initial_ms[(param_start+4)*d:-1]  + 2) - 1)  
            initial_ms_map[-1] = np.log(initial_ms[-1]- 2)
            if self.dist == 'norm':
                initial_ms_map = initial_ms_map[:-1]
                
            return initial_ms_map
        
    def fit(self):
        if self.dist == 't':
            res = minimize(self.loglik_stu, self.get_init_params())

        elif self.dist == 'norm':
            res = minimize(self.loglik_norm, self.get_init_params())

        return res

    def loglik_stu(self,theta):
        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''

        T, d = self.ret.shape

        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = exp(-theta[param_start*d:(param_start+1)*d])

        A1 = diag(exp(-theta[(param_start+1)*d:(param_start+2)*d]))

        B1 = diag(1/(1+exp(-theta[(param_start+2)*d:(param_start+3)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+3)*d:(param_start+4)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+4)*d:(param_start+4)*d+off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]

        nu = exp(theta[-1])+2

        h1 = zeros([d,T])

        h1[:,0] = std(eps1,0)

        loglik = zeros((T,1))

        eps1_t = eps1.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t

        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1 - A1_dot_G_dot_eps1_t

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1

        L1 = np.linalg.cholesky(sigma1)
        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L1_density =  L1_mahalanobis_squared/L1_diag


        loglik = np.sum(np.log(L1_density[1:]) + const)
        L = -loglik/T
        self.loglik = loglik
        return L 

    def loglik_norm(self,theta):
        '''
        Function estimating markov switching norm multivariate case in 2 regimes
        '''

        T, d = self.ret.shape
    
        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = exp(-theta[param_start*d:(param_start+1)*d])

        A1 = diag(exp(-theta[(param_start+1)*d:(param_start+2)*d]))

        B1 = diag(1/(1+exp(-theta[(param_start+2)*d:(param_start+3)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+3)*d:(param_start+4)*d])))
        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+4)*d:(param_start+4)*d+off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]

        # volatility
        h1 = zeros([d,T])

        h1[:,0]= std(eps1,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A1_dot_G_dot_eps1_t = A1@G@eps1_t

        const = np.power((2.0 * np.pi), d / 2.0)
        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1 - A1_dot_G_dot_eps1_t

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1

        L1 = np.linalg.cholesky(sigma1)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L1_mahalanobis_squared = exp(-0.5 *np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1))
        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L1_density =  1.0 / (const* L1_diag) * L1_mahalanobis_squared

        loglik = np.sum(np.log(L1_density[1:]))
        L = -loglik/T
        self.loglik = loglik
        return L 
    

    def get_params(self,theta):
        T, d = self.ret.shape

        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = exp(-theta[param_start*d:(param_start+1)*d])

        A1 = diag(exp(-theta[(param_start+1)*d:(param_start+2)*d]))

        B1 = diag(1/(1+exp(-theta[(param_start+2)*d:(param_start+3)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+3)*d:(param_start+4)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+4)*d:(param_start+4)*d+off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]

        nu = exp(theta[-1])+2
        if self.dist == 't':  
            return  np.concatenate((mu,w1,diag(A1),diag(B1),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],np.array([nu]))).ravel()
        else:
            return  np.concatenate((mu,w1,diag(A1),diag(B1),diag(G),R1[np.triu_indices(R1.shape[0], k=1)])).ravel()

    def normsinglefore(self,theta):
        '''
        Function estimating markov switching norm multivariate case in 2 regimes
        '''

        T, d = self.ret.shape

        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = exp(-theta[param_start*d:(param_start+1)*d])

        A1 = diag(exp(-theta[(param_start+1)*d:(param_start+2)*d]))

        B1 = diag(1/(1+exp(-theta[(param_start+2)*d:(param_start+3)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+3)*d:(param_start+4)*d])))
        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+4)*d:(param_start+4)*d+off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]

        # volatility
        h1 = zeros([d,T])

        h1[:,0]= std(eps1,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T

        abs_eps1 = abs(eps1_t)

        A1_dot_abs_eps1 = A1@abs_eps1

        A1_dot_G_dot_eps1_t = A1@G@eps1_t


        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]


        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1

        L1 = np.linalg.cholesky(sigma1)

        const = np.power((2.0 * np.pi), d / 2.0)
        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)

            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
        
            loglik[t,0] =np.log(1.0 / (const* prod(np.diag(L1[t]))) * exp(-0.5 * L1_mahalanobis_squared))
            
        return h1[:,-1],mu, w1.reshape(-1, 1) , A1, B1, G, R[0]

    def stusinglefore(self,theta):
        T, d = self.ret.shape

        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = exp(-theta[param_start*d:(param_start+1)*d])

        A1 = diag(exp(-theta[(param_start+1)*d:(param_start+2)*d]))

        B1 = diag(1/(1+exp(-theta[(param_start+2)*d:(param_start+3)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+3)*d:(param_start+4)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+4)*d:(param_start+4)*d+off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]

        nu = exp(theta[-1])+2

        h1 = zeros([d,T])

        h1[:,0] = std(eps1,0)

        loglik = zeros((T,1))

        eps1_t = eps1.T

        abs_eps1 = abs(eps1_t)

        A1_dot_abs_eps1 = A1@abs_eps1

        A1_dot_G_dot_eps1_t = A1@G@eps1_t

        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)

        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
    
        sigma1 = expanded_h1@R1@expanded_h1
     
        L1 = np.linalg.cholesky(sigma1)

        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
   
            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu

            loglik[t,0] =np.log((1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1[t]))) + const

        return h1[:,-1], mu, w1.reshape(-1, 1), A1, B1, G, R[0], nu
    
    def forecast_single_norm(self,w, A, B, G, R, sinit, D):
        T, n = self.ret.shape
        Atilde = A * G
        Rtilde = 2 / np.pi * (R * np.arcsin(R) + np.sqrt(1 - R ** 2))
        kappa1 = np.sqrt(2 / np.pi)
        Y = np.concatenate([sinit, np.ravel(np.outer(sinit, sinit))]).reshape(-1, 1) 

        C11 = kappa1 * A + B
        C21 = np.kron(w, C11) + np.kron(C11, w)
        C22 = np.kron(B, B) + kappa1 * (np.kron(B, A) + np.kron(A, B)) + np.kron(A, A) @ np.diag(np.ravel(Rtilde)) + \
            np.kron(Atilde, Atilde) @ np.diag(np.ravel(R))
        Ctilde = np.block([[C11,np.zeros((self.k*n,self.k**2 * n**2))], [C21, C22]])
        
        wtilde = np.concatenate([w, np.kron(w, w)])

        sigmafore = np.zeros((n, n, D))
        sigmafore[:, :, 0] = R * np.reshape(Y[n:], (n, n))

        for d in range(1, D):
            Y = wtilde + Ctilde @ Y
            sigmafore[:, :, d] = R * np.reshape(Y[n:], (n, n))

        sigmalong = np.sum(sigmafore, axis=2)
        
        return sigmalong
    
    
    def forecast_single_stu(self,w, A, B, G, R, sinit,nu, D):
        T, n = self.ret.shape
        Atilde = A * G
        Rtilde = 2 / np.pi * (R * np.arcsin(R) + np.sqrt(1 - R ** 2))
        kappa1 = np.sqrt(nu-2)/np.sqrt(np.pi)*exp(scipy.special.loggamma((nu-1)/2)-scipy.special.loggamma(nu/2))
        Y = np.concatenate([sinit, np.ravel(np.outer(sinit, sinit))]).reshape(-1, 1) 

        C11 = kappa1 * A + B
        C21 = np.kron(w, C11) + np.kron(C11, w)
        C22 = np.kron(B, B) + kappa1 * (np.kron(B, A) + np.kron(A, B)) + np.kron(A, A) @ np.diag(np.ravel(Rtilde)) + \
            np.kron(Atilde, Atilde) @ np.diag(np.ravel(R))
        Ctilde = np.block([[C11, np.zeros((self.k*n,self.k * n**2))], [C21, C22]])
        wtilde = np.concatenate([w, np.kron(w, w)])

        sigmafore = np.zeros((n, n, D))
        sigmafore[:, :, 0] = R * np.reshape(Y[n:], (n, n))

        for d in range(1, D):
            Y = wtilde + Ctilde @ Y
            sigmafore[:, :, d] = R * np.reshape(Y[n:], (n, n))

        sigmalong = np.sum(sigmafore, axis=2)
        
        
        return sigmalong

    
    def norm_err(self,theta):
            
        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''

        T, d = self.ret.shape

        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = theta[param_start*d:(param_start+1)*d]

        A1 = diag(theta[(param_start+1)*d:(param_start+2)*d])

        B1 = diag(theta[(param_start+2)*d:(param_start+3)*d])

        G = diag(theta[(param_start+3)*d:(param_start+4)*d])

        R1 = self.fill_correlation_matrix(theta[(param_start+4)*d:])

        h1 = zeros([d,T])

        h1[:,0] = std(eps1,0)

        loglik = zeros((T,1))

        eps1_t = eps1.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t

        const = np.power((2.0 * np.pi), d / 2.0)
        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1 - A1_dot_G_dot_eps1_t

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1

        L1 = np.linalg.cholesky(sigma1)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L1_mahalanobis_squared = exp(-0.5 *np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1))
        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L1_density =  1.0 / (const* L1_diag) * L1_mahalanobis_squared

        loglik = np.sum(np.log(L1_density[1:]))
        return loglik
    
    def stu_err(self,theta):

        '''
        Function estimating markov switching student t multivariate case in 2 regimes
        '''

        T, d = self.ret.shape


        param_start = 1
        mu = theta[0:d]
        eps1 = self.ret - mu

        w1 = theta[param_start*d:(param_start+1)*d]

        A1 = diag(theta[(param_start+1)*d:(param_start+2)*d])
        B1 = diag(theta[(param_start+2)*d:(param_start+3)*d])

        G = diag(theta[(param_start+3)*d:(param_start+4)*d])

        R1 = self.fill_correlation_matrix(theta[(param_start+4)*d:-1])

        nu = theta[-1]

        h1 = zeros([d,T])

        h1[:,0] = std(eps1,0)

        loglik = zeros((T,1))

        eps1_t = eps1.T

        abs_eps1 = abs(eps1_t)

        A1_dot_abs_eps1 = A1@abs_eps1

        A1_dot_G_dot_eps1_t = A1@G@eps1_t

        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1 - A1_dot_G_dot_eps1_t

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1

        L1 = np.linalg.cholesky(sigma1)
        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L1_density =  L1_mahalanobis_squared/L1_diag


        loglik = np.sum(np.log(L1_density[1:]) + const)
        return loglik

            
