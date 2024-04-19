import numpy as np
import scipy
import scipy.linalg.lapack as lapack
from numpy import array, diag, exp, log, prod, sqrt, std, zeros
from scipy.linalg import expm, norm
from scipy.optimize import minimize
from scipy.special import gamma, kn, kv

class mgarch_3_regimes:
    """
    Implements the Multivariate Markov Switching CCC GARCH model of Markus Haas & Ji-Chun Liu [1] for three regimes.
    Please find description of main changes in ms_ccc_garch_2.py"""
    def __init__(self,ret,dist='norm', regime_mean=False,k =3,init_params = np.empty(0)):
        self.regime_mean = regime_mean
        self.ret = ret 
        self.k = k 
        self.init_params = init_params
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
    
    def get_init_params(self):

        if (self.init_params== None).all() == False:
            return self.init_params
        else:
            if self.regime_mean == True:
                param_start = 3
            else:
                param_start = 1

            T, d = self.ret.shape
            off_diag = int(d * (d - 1) /2 )

            initial_ms = np.random.uniform(0.05, 0.7, size=(param_start*d + 3*d*3 + d + off_diag*self.k,))
            initial_ms = np.append(initial_ms,[0.9,0.9,0.9,0.9,0.9,0.9,8])
            initial_ms_map = np.zeros(len(initial_ms))
            initial_ms_map[param_start*d:(param_start+6)*d] = -np.log(initial_ms[param_start*d:(param_start+6)*d] )
            initial_ms_map[(param_start+6)*d:(param_start+9)*d] = np.log(initial_ms[(param_start+6)*d:(param_start+9)*d]/(1-initial_ms[(param_start+6)*d:(param_start+9)*d]))
            initial_ms_map[(param_start+9)*d:(param_start+10)*d]  = np.log((1+initial_ms[(param_start+8)*d:(param_start+9)*d])/(1-initial_ms[(param_start+9)*d:(param_start+10)*d]))
            initial_ms_map[(param_start+10)*d:-7]  = -log(4/(initial_ms[(param_start+10)*d:-7]  + 2) - 1)

            initial_ms_map[-7:-1] = np.log(initial_ms[-7:-1]/(1-initial_ms[-7:-1]))
            initial_ms_map[-1] = np.log(initial_ms[-1]- 2)
            if self.dist == 'norm':
                initial_ms_map = initial_ms_map[:-1]
                
            return initial_ms_map


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
    

    def cov_to_corr(self,cov_matrix):
        n = cov_matrix.shape[0]
        corr_matrix = np.zeros_like(cov_matrix)
        
        for i in range(n):
            for j in range(n):
                corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                
        return corr_matrix

    def loglik_stu(self,theta):
        '''
        Log likelihood estimation of multivariate MS-CCC with student t innovations with 2 regimes
        '''

        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = exp(-theta[param_start*d:(param_start+1)*d]),exp(-theta[(param_start+1)*d:(param_start+2)*d]),exp(-theta[(param_start+2)*d:(param_start+3)*d])

        A1,A2,A3 = diag(exp(-theta[(param_start+3)*d:(param_start+4)*d])),diag(exp(-theta[(param_start+4)*d:(param_start+5)*d])),diag(exp(-theta[(param_start+5)*d:(param_start+6)*d]))

        B1,B2,B3 = diag(1/(1+exp(-theta[(param_start+6)*d:(param_start+7)*d]))),diag(1/(1+exp(-theta[(param_start+7)*d:(param_start+8)*d]))),diag(1/(1+exp(-theta[(param_start+8)*d:(param_start+9)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+9)*d:(param_start+10)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+10)*d:(param_start+10)*d+off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]
        R3 = R[2]

        P = np.ones((2,3))
        P[:,0] = np.exp(theta[-7:-5])/(1+np.exp(theta[-7])+np.exp(theta[-6]))
        P[:,1] = np.exp(theta[-5:-3])/(1+np.exp(theta[-5])+np.exp(theta[-4]))
        P[:,2] = np.exp(theta[-3:-1])/(1+np.exp(theta[-3])+np.exp(theta[-2]))
        P = np.vstack([P, 1 - sum(P)])
        A = np.eye(3)-P
        A = np.vstack([A,np.ones((1,3))])


        pinf = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array([0,0,0,1]))

        ksi = pinf
        nu = exp(theta[-1])+2

        h1,h2,h3 = zeros([d,T]),zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0],h3[:,0]  = std(eps1,0),std(eps2,0),std(eps3,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        eps3_t = eps3.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A3_dot_abs_eps3 = A3@abs(eps3_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        A3_dot_G_dot_eps3_t = A3@G@eps3_t


        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        LL = np.zeros((3,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]
        w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t = w3.T[:,None]  + A3_dot_abs_eps3[:,:] - A3_dot_G_dot_eps3_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]
            h3[:,t] = w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t[:,t-1] + B3@h3[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        expanded_h3 = np.eye(d) * np.expand_dims(h3.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        sigma3 = expanded_h3@R3@expanded_h3

        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        L3 = np.linalg.cholesky(sigma3)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)
        L3_inv_y_minus_mu = np.linalg.solve(L3[:], eps3_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L2_mahalanobis_squared = (1 + np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L3_mahalanobis_squared = (1 + np.sum(L3_inv_y_minus_mu * L3_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)


        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)
        L3_diag = np.prod(np.diagonal(L3, axis1=1, axis2=2),axis = 1)

        L1_density =  L1_mahalanobis_squared/L1_diag
        L2_density =  L2_mahalanobis_squared/L2_diag
        L3_density =  L3_mahalanobis_squared/L3_diag


        for t in range(1,T):
            LL[0,0] = ksi[0].item()*L1_density[t]
            LL[1,0] = ksi[1].item()*L2_density[t]
            LL[2,0] = ksi[2].item()*L3_density[t]

            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] =np.log(np.dot(np.ones((1,3)),LL)) + const
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        L = -loglik/T
        return L 

    def loglik_norm(self,theta):
        '''
        Log likelihood estimation of multivariate MS-CCC with student t innovations with 2 regimes
        '''

        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = exp(-theta[param_start*d:(param_start+1)*d]),exp(-theta[(param_start+1)*d:(param_start+2)*d]),exp(-theta[(param_start+2)*d:(param_start+3)*d])

        A1,A2,A3 = diag(exp(-theta[(param_start+3)*d:(param_start+4)*d])),diag(exp(-theta[(param_start+4)*d:(param_start+5)*d])),diag(exp(-theta[(param_start+5)*d:(param_start+6)*d]))

        B1,B2,B3 = diag(1/(1+exp(-theta[(param_start+6)*d:(param_start+7)*d]))),diag(1/(1+exp(-theta[(param_start+7)*d:(param_start+8)*d]))),diag(1/(1+exp(-theta[(param_start+8)*d:(param_start+9)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+9)*d:(param_start+10)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+10)*d:(param_start+10)*d+off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]
        R3 = R[2]

        P = np.ones((2,3))
        P[:,0] = np.exp(theta[-7:-5])/(1+np.exp(theta[-7])+np.exp(theta[-6]))
        P[:,1] = np.exp(theta[-5:-3])/(1+np.exp(theta[-5])+np.exp(theta[-4]))
        P[:,2] = np.exp(theta[-3:-1])/(1+np.exp(theta[-3])+np.exp(theta[-2]))
        P = np.vstack([P, 1 - sum(P)])
        A = np.eye(3)-P
        A = np.vstack([A,np.ones((1,3))])


        pinf = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array([0,0,0,1]))

        ksi = pinf
        
        h1,h2,h3 = zeros([d,T]),zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0],h3[:,0]  = std(eps1,0),std(eps2,0),std(eps3,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        eps3_t = eps3.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A3_dot_abs_eps3 = A3@abs(eps3_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        A3_dot_G_dot_eps3_t = A3@G@eps3_t


        const = np.power((2.0 * np.pi), d / 2.0)
        LL = np.zeros((3,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]
        w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t = w3.T[:,None]  + A3_dot_abs_eps3[:,:] - A3_dot_G_dot_eps3_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]
            h3[:,t] = w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t[:,t-1] + B3@h3[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        expanded_h3 = np.eye(d) * np.expand_dims(h3.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        sigma3 = expanded_h3@R3@expanded_h3

        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        L3 = np.linalg.cholesky(sigma3)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)
        L3_inv_y_minus_mu = np.linalg.solve(L3[:], eps3_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = exp(-0.5 *np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1))
        L2_mahalanobis_squared = exp(-0.5 *np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1))
        L3_mahalanobis_squared = exp(-0.5 *np.sum(L3_inv_y_minus_mu * L3_inv_y_minus_mu, axis=1))


        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)
        L3_diag = np.prod(np.diagonal(L3, axis1=1, axis2=2),axis = 1)

        L1_density =  1.0 / (const* L1_diag) * L1_mahalanobis_squared
        L2_density =  1.0 / (const* L2_diag) * L2_mahalanobis_squared
        L3_density =  1.0 / (const* L3_diag) * L3_mahalanobis_squared

            
        for t in range(1,T):
            LL[0,0] = ksi[0].item()*L1_density[t]
            LL[1,0] = ksi[1].item()*L2_density[t]
            LL[2,0] = ksi[2].item()*L3_density[t]
            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] =np.log(np.dot(np.ones((1,3)),LL))
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        L = -loglik/T
        return L 

    
    def smooth_prob_log_stu_b_m(self,theta):
        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = exp(-theta[param_start*d:(param_start+1)*d]),exp(-theta[(param_start+1)*d:(param_start+2)*d]),exp(-theta[(param_start+2)*d:(param_start+3)*d])

        A1,A2,A3 = diag(exp(-theta[(param_start+3)*d:(param_start+4)*d])),diag(exp(-theta[(param_start+4)*d:(param_start+5)*d])),diag(exp(-theta[(param_start+5)*d:(param_start+6)*d]))

        B1,B2,B3 = diag(1/(1+exp(-theta[(param_start+6)*d:(param_start+7)*d]))),diag(1/(1+exp(-theta[(param_start+7)*d:(param_start+8)*d]))),diag(1/(1+exp(-theta[(param_start+8)*d:(param_start+9)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+9)*d:(param_start+10)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+10)*d:(param_start+10)*d+off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]
        R3 = R[2]

        P = np.ones((2,3))
        P[:,0] = np.exp(theta[-7:-5])/(1+np.exp(theta[-7])+np.exp(theta[-6]))
        P[:,1] = np.exp(theta[-5:-3])/(1+np.exp(theta[-5])+np.exp(theta[-4]))
        P[:,2] = np.exp(theta[-3:-1])/(1+np.exp(theta[-3])+np.exp(theta[-2]))
        P = np.vstack([P, 1 - sum(P)])
        A = np.eye(3)-P
        A = np.vstack([A,np.ones((1,3))])


        pinf = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array([0,0,0,1]))

        ksi=np.zeros([self.k ,T+1])
        xsi=np.zeros([self.k ,T])
        ksi[:,1] = pinf.T

        nu = exp(theta[-1])+2

        h1,h2,h3 = zeros([d,T]),zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0],h3[:,0]  = std(eps1,0),std(eps2,0),std(eps3,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        eps3_t = eps3.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A3_dot_abs_eps3 = A3@abs(eps3_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        A3_dot_G_dot_eps3_t = A3@G@eps3_t


        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        LL = np.zeros((self.k,1))

        condcorr = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)

        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]
            h3[:,t] = w3 + A3_dot_abs_eps3[:,t-1]-A3_dot_G_dot_eps3_t[:,t-1] + B3@h3[:,t-1]


        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        expanded_h3 = np.eye(d) * np.expand_dims(h3.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        sigma3 = expanded_h3@R3@expanded_h3

        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        L3 = np.linalg.cholesky(sigma3)

        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
            L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
            L3_inv_y_minus_mu, _  = lapack.dtrtrs(L3[t], eps3_t[:,t], lower=True)

            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
            L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
            L3_mahalanobis_squared = L3_inv_y_minus_mu.T @ L3_inv_y_minus_mu


            LL[0,0] = ksi[0,t]*(1+L1_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L1[t]))
            LL[1,0] = ksi[1,t]*(1+L2_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L2[t]))
            LL[2,0] = ksi[2,t]*(1+L3_mahalanobis_squared/(nu-2))**(-(nu+d)/2)/prod(np.diag(L3[t]))

            fporb =LL/sum(LL)
            xsi[0,t] = fporb[0].item()
            xsi[1,t] = fporb[1].item()
            xsi[2,t] = fporb[2].item()
            ksi[:,t+1] =np.dot(P,xsi[:,t])
            loglik[t,0] =np.log(np.dot(np.ones((1,self.k )),LL)) + const
            sigmabar = ksi[0,t] *sigma1[t] + ksi[1,t] *sigma2[t] + ksi[2,t] *sigma3[t]
            condcorr[t] = self.cov_to_corr(sigmabar)

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])

        chi = np.zeros([self.k ,T])
        chi[:,T-1] = xsi[:,T-1]

        for t in range(1,T):
            chi[:,T-t-1]=xsi[:,T-t-1]*(P.T@(chi[:,T-t]/ksi[:,T-t]))

        if self.regime_mean == True:
            est = [mu,mu2,mu3,w1,w2,w3, np.diag(A1), np.diag(A2),np.diag(A3), np.diag(B1), np.diag(B2), np.diag(B3), np.diag(G),R,P,nu]
        else:
            est = [mu,w1,w2,w3, np.diag(A1), np.diag(A2),np.diag(A3), np.diag(B1), np.diag(B2), np.diag(B3), np.diag(G),R,P,nu]
        return xsi,chi,condcorr,est,h1,h2,h3,loglik

    def smooth_prob_norm(self,theta):
        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = exp(-theta[param_start*d:(param_start+1)*d]),exp(-theta[(param_start+1)*d:(param_start+2)*d]),exp(-theta[(param_start+2)*d:(param_start+3)*d])

        A1,A2,A3 = diag(exp(-theta[(param_start+3)*d:(param_start+4)*d])),diag(exp(-theta[(param_start+4)*d:(param_start+5)*d])),diag(exp(-theta[(param_start+5)*d:(param_start+6)*d]))

        B1,B2,B3 = diag(1/(1+exp(-theta[(param_start+6)*d:(param_start+7)*d]))),diag(1/(1+exp(-theta[(param_start+7)*d:(param_start+8)*d]))),diag(1/(1+exp(-theta[(param_start+8)*d:(param_start+9)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+9)*d:(param_start+10)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+10)*d:(param_start+10)*d+off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]
        R3 = R[2]

        P = np.ones((2,3))
        P[:,0] = np.exp(theta[-7:-5])/(1+np.exp(theta[-7])+np.exp(theta[-6]))
        P[:,1] = np.exp(theta[-5:-3])/(1+np.exp(theta[-5])+np.exp(theta[-4]))
        P[:,2] = np.exp(theta[-3:-1])/(1+np.exp(theta[-3])+np.exp(theta[-2]))
        P = np.vstack([P, 1 - sum(P)])
        A = np.eye(3)-P
        A = np.vstack([A,np.ones((1,3))])


        pinf = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array([0,0,0,1]))

        ksi=np.zeros([self.k ,T+1])
        
        xsi=np.zeros([self.k ,T])
        ksi[:,1] = pinf.T
        
        h1,h2,h3 = zeros([d,T]),zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0],h3[:,0]  = std(eps1,0),std(eps2,0),std(eps3,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        eps3_t = eps3.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A3_dot_abs_eps3 = A3@abs(eps3_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        A3_dot_G_dot_eps3_t = A3@G@eps3_t


        const = np.power((2.0 * np.pi), d / 2.0)
        LL = np.zeros((3,1))


        condcorr = np.eye(d) * np.expand_dims(zeros([d,T+1]).T, axis=2)

        for t in range(1,T):
            h1[:,t] = w1 + A1_dot_abs_eps1[:,t-1]-A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2 + A2_dot_abs_eps2[:,t-1]-A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]
            h3[:,t] = w3 + A3_dot_abs_eps3[:,t-1]-A3_dot_G_dot_eps3_t[:,t-1] + B3@h3[:,t-1]


        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        expanded_h3 = np.eye(d) * np.expand_dims(h3.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        sigma3 = expanded_h3@R3@expanded_h3

        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        L3 = np.linalg.cholesky(sigma3)

        for t in range(1,T):
            L1_inv_y_minus_mu, _  = lapack.dtrtrs(L1[t], eps1_t[:,t], lower=True)
            L2_inv_y_minus_mu, _  = lapack.dtrtrs(L2[t], eps2_t[:,t], lower=True)
            L3_inv_y_minus_mu, _  = lapack.dtrtrs(L3[t], eps3_t[:,t], lower=True)

            L1_mahalanobis_squared = L1_inv_y_minus_mu.T @ L1_inv_y_minus_mu
            L2_mahalanobis_squared = L2_inv_y_minus_mu.T @ L2_inv_y_minus_mu
            L3_mahalanobis_squared = L3_inv_y_minus_mu.T @ L3_inv_y_minus_mu


            LL[0,0] = ksi[0,t]*(1.0 / (const* prod(np.diag(L1[t]))) * exp(-0.5 * L1_mahalanobis_squared))
            LL[1,0] = ksi[1,t]*(1.0 / (const* prod(np.diag(L2[t]))) * exp(-0.5 * L2_mahalanobis_squared))
            LL[2,0] = ksi[2,t]*(1.0 / (const* prod(np.diag(L3[t]))) * exp(-0.5 * L3_mahalanobis_squared))

            fporb =LL/sum(LL)
            xsi[0,t] = fporb[0].item()
            xsi[1,t] = fporb[1].item()
            xsi[2,t] = fporb[2].item()
            ksi[:,t+1] =np.dot(P,xsi[:,t])
            loglik[t,0] =np.log(np.dot(np.ones((1,self.k )),LL)) 
            sigmabar = ksi[0,t] *sigma1[t] + ksi[1,t] *sigma2[t] + ksi[2,t] *sigma3[t]
            condcorr[t] = self.cov_to_corr(sigmabar)

        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])

        chi = np.zeros([self.k ,T])
        chi[:,T-1] = xsi[:,T-1]

        for t in range(1,T):
            chi[:,T-t-1]=xsi[:,T-t-1]*(P.T@(chi[:,T-t]/ksi[:,T-t]))

        if self.regime_mean == True:
            est = [mu,mu2,mu3,w1,w2,w3, np.diag(A1), np.diag(A2),np.diag(A3), np.diag(B1), np.diag(B2), np.diag(B3), np.diag(G),R,P]
        else:
            est = [mu,w1,w2,w3, np.diag(A1), np.diag(A2),np.diag(A3), np.diag(B1), np.diag(B2), np.diag(B3), np.diag(G),R,P]
        return xsi,chi,condcorr,est,h1,h2,h3,loglik
    

    def fit(self):
        if self.dist == 't':
            res = minimize(self.loglik_stu, self.get_init_params())

        elif self.dist == 'norm':
            res = minimize(self.loglik_norm, self.get_init_params())

        return res
    
    def get_smoothed_prob(self,params):
        if self.dist == 't':
            xsi,chi,condcorr,est,h1,h2,h3,loglik = self.smooth_prob_log_stu_b_m(params)

        elif self.dist == 'norm':
            xsi,chi,condcorr,est,h1,h2,h3,loglik = self.smooth_prob_norm(params)
            
        return xsi,chi,condcorr,est,h1,h2,h3,loglik

    def get_params(self,theta):
        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = exp(-theta[param_start*d:(param_start+1)*d]),exp(-theta[(param_start+1)*d:(param_start+2)*d]),exp(-theta[(param_start+2)*d:(param_start+3)*d])

        A1,A2,A3 = diag(exp(-theta[(param_start+3)*d:(param_start+4)*d])),diag(exp(-theta[(param_start+4)*d:(param_start+5)*d])),diag(exp(-theta[(param_start+5)*d:(param_start+6)*d]))

        B1,B2,B3 = diag(1/(1+exp(-theta[(param_start+6)*d:(param_start+7)*d]))),diag(1/(1+exp(-theta[(param_start+7)*d:(param_start+8)*d]))),diag(1/(1+exp(-theta[(param_start+8)*d:(param_start+9)*d])))

        G = diag(-1+2/(1+np.exp(-theta[(param_start+9)*d:(param_start+10)*d])))

        
        off_diag = int(d * (d - 1) /2 )
        rho = [-2 + 4/(1+exp(-theta[(param_start+10)*d:(param_start+10)*d+off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag])),-2 + 4/(1+exp(-theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]))]
        
        R = [self.cov_matrix_inverse_matrix(rho_vec,1e-10) for rho_vec in rho]
        R1 = R[0]
        R2 = R[1]
        R3 = R[2]

        P = np.ones((2,3))
        P0 = np.exp(theta[-7:-5])/(1+np.exp(theta[-7])+np.exp(theta[-6]))
        P1 = np.exp(theta[-5:-3])/(1+np.exp(theta[-5])+np.exp(theta[-4]))
        P2 = np.exp(theta[-3:-1])/(1+np.exp(theta[-3])+np.exp(theta[-2]))

        if self.dist == 't':  
            p,q = 1/(1+exp(-theta[-3])),1/(1+exp(-theta[-2]))
            nu = exp(theta[-1])+2
            return  np.concatenate((mu,w1,w2,w3,diag(A1),diag(A2),diag(A3),diag(B1),diag(B2),diag(B3),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],R2[np.triu_indices(R2.shape[0], k=1)],R3[np.triu_indices(R3.shape[0], k=1)],np.array(P0),np.array(P1),np.array(P2),np.array([nu]))).ravel()
        else:
            p,q = 1/(1+exp(-theta[-2])),1/(1+exp(-theta[-1]))
            return  np.concatenate((mu,w1,w2,w3,diag(A1),diag(A2),diag(A3),diag(B1),diag(B2),diag(B3),diag(G),R1[np.triu_indices(R1.shape[0], k=1)],R2[np.triu_indices(R2.shape[0], k=1)],R3[np.triu_indices(R3.shape[0], k=1)],np.array(P0),np.array(P1),np.array(P2))).ravel()
        
    def stu_err(self,theta):
        '''
        Log likelihood estimation of multivariate MS-CCC with student t innovations with 2 regimes
        '''

        T, d = self.ret.shape

        if self.regime_mean == True:
            param_start = 3
            mu,mu2,mu3 = theta[0:d],theta[d:2*d],theta[2*d:3*d]
            eps1 = self.ret - mu                                                                                                                         
            eps2 = self.ret - mu2
            eps3 = self.ret - mu3
        else:
            param_start = 1
            mu = theta[0:d]
            eps1 = self.ret - mu
            eps2 = self.ret - mu
            eps3 = self.ret - mu

        w1,w2,w3 = theta[param_start*d:(param_start+1)*d],theta[(param_start+1)*d:(param_start+2)*d],theta[(param_start+2)*d:(param_start+3)*d]

        A1,A2,A3 = diag(theta[(param_start+3)*d:(param_start+4)*d]),diag(theta[(param_start+4)*d:(param_start+5)*d]),diag(theta[(param_start+5)*d:(param_start+6)*d])

        B1,B2,B3 = diag(theta[(param_start+6)*d:(param_start+7)*d]),diag(theta[(param_start+7)*d:(param_start+8)*d]),diag(theta[(param_start+8)*d:(param_start+9)*d])

        G = diag(theta[(param_start+9)*d:(param_start+10)*d])

        
        off_diag = int(d * (d - 1) /2 )
        rho = [theta[(param_start+10)*d:(param_start+10)*d+off_diag],theta[(param_start+10)*d+off_diag:(param_start+10)*d+2*off_diag],theta[(param_start+10)*d+2*off_diag:(param_start+10)*d+3*off_diag]]
        
        R1 = self.fill_correlation_matrix(rho[0])
        R2 = self.fill_correlation_matrix(rho[1])
        R3 = self.fill_correlation_matrix(rho[2])

        P = np.ones((2,3))
        P[:,0] = theta[-7:-5]
        P[:,1] = theta[-5:-3]
        P[:,2] = theta[-3:-1]
        P = np.vstack([P, 1 - sum(P)])
        A = np.eye(3)-P
        A = np.vstack([A,np.ones((1,3))])


        pinf = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),np.array([0,0,0,1]))

        ksi = pinf
        nu = theta[-1]

        h1,h2,h3 = zeros([d,T]),zeros([d,T]),zeros([d,T])

        h1[:,0],h2[:,0],h3[:,0]  = std(eps1,0),std(eps2,0),std(eps3,0)

        loglik = zeros((T,1))
        eps1_t = eps1.T
        eps2_t = eps2.T
        eps3_t = eps3.T

        A1_dot_abs_eps1 = A1@abs(eps1_t)
        A2_dot_abs_eps2 = A2@abs(eps2_t)
        A3_dot_abs_eps3 = A3@abs(eps3_t)

        A1_dot_G_dot_eps1_t = A1@G@eps1_t
        A2_dot_G_dot_eps2_t = A2@G@eps2_t
        A3_dot_G_dot_eps3_t = A3@G@eps3_t


        const =  scipy.special.loggamma(.5*(nu+d))-scipy.special.loggamma(nu/2)-d/2*np.log(np.pi)-d/2*np.log(nu-2)
        LL = np.zeros((3,1))

        w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t = w1.T[:,None]  + A1_dot_abs_eps1[:,:] - A1_dot_G_dot_eps1_t[:,:]
        w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t = w2.T[:,None]  + A2_dot_abs_eps2[:,:] - A2_dot_G_dot_eps2_t[:,:]
        w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t = w3.T[:,None]  + A3_dot_abs_eps3[:,:] - A3_dot_G_dot_eps3_t[:,:]

        for t in range(1,T):
            h1[:,t] = w1_A1_dot_abs_eps1_A1_dot_G_dot_eps1_t[:,t-1] + B1@h1[:,t-1]
            h2[:,t] = w2_A2_dot_abs_eps2_A2_dot_G_dot_eps2_t[:,t-1] + B2@h2[:,t-1]
            h3[:,t] = w3_A3_dot_abs_eps3_A3_dot_G_dot_eps3_t[:,t-1] + B3@h3[:,t-1]

        expanded_h1 = np.eye(d) * np.expand_dims(h1.T, axis=2)
        expanded_h2 = np.eye(d) * np.expand_dims(h2.T, axis=2)
        expanded_h3 = np.eye(d) * np.expand_dims(h3.T, axis=2)

        sigma1 = expanded_h1@R1@expanded_h1
        sigma2 = expanded_h2@R2@expanded_h2
        sigma3 = expanded_h3@R3@expanded_h3

        L1 = np.linalg.cholesky(sigma1)
        L2 = np.linalg.cholesky(sigma2)
        L3 = np.linalg.cholesky(sigma3)

        L1_inv_y_minus_mu = np.linalg.solve(L1[:], eps1_t[:, :].T[:,:,None]).squeeze(-1)
        L2_inv_y_minus_mu = np.linalg.solve(L2[:], eps2_t[:, :].T[:,:,None]).squeeze(-1)
        L3_inv_y_minus_mu = np.linalg.solve(L3[:], eps3_t[:, :].T[:,:,None]).squeeze(-1)

        L1_mahalanobis_squared = (1 + np.sum(L1_inv_y_minus_mu * L1_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L2_mahalanobis_squared = (1 + np.sum(L2_inv_y_minus_mu * L2_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)
        L3_mahalanobis_squared = (1 + np.sum(L3_inv_y_minus_mu * L3_inv_y_minus_mu, axis=1)/(nu-2))**(-(nu+d)/2)


        L1_diag = np.prod(np.diagonal(L1, axis1=1, axis2=2),axis = 1)
        L2_diag = np.prod(np.diagonal(L2, axis1=1, axis2=2),axis = 1)
        L3_diag = np.prod(np.diagonal(L3, axis1=1, axis2=2),axis = 1)

        L1_density =  L1_mahalanobis_squared/L1_diag
        L2_density =  L2_mahalanobis_squared/L2_diag
        L3_density =  L3_mahalanobis_squared/L3_diag


        for t in range(1,T):
            LL[0,0] = ksi[0].item()*L1_density[t]
            LL[1,0] = ksi[1].item()*L2_density[t]
            LL[2,0] = ksi[2].item()*L3_density[t]

            xsi = LL/np.sum(LL)
            ksi=np.dot(P,xsi)
            loglik[t,0] =np.log(np.dot(np.ones((1,3)),LL)) + const
            
        loglik = np.dot(np.ones((1,T-1)),loglik[1:T])
        return loglik 