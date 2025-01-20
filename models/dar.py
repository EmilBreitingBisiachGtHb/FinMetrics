import numpy as np
import scipy.optimize
from scipy.special import loggamma
from scipy.stats import norm as gaussian, t as student_t
import numpy as np
from numpy.linalg import eigvals
import torch
torch.set_printoptions(precision=8)

import os
import sys
sys.path.append(os.path.abspath(".."))
from utils import ParamTracker

class DAR:
    def __init__(self, p = 1, dist = "gaussian"):
        
        self.p = p
        if not isinstance(p, int) or p <= 0:
            raise ValueError("'p' must be a strictly positive integer.")
        
        self.dist = dist
        if dist not in ["student-t", "gaussian"]:
            raise ValueError("'dist' must be either 'student-t' or 'gaussian'.")
        
        self.params = {
            "rho": {"init": np.array([0.2] * p), "bound": [(None, None)] * p},
            "alpha": {"init": np.array([0.5] * p), "bound": [(0.001, None)] * p},
            "omega": {"init": np.array([0.5]), "bound": [(0.001, None)]},
            "nu": {"init": np.array([5]), "bound": [(2.001, None)]},
            }
        
        self.T_sim = 100000
        self.max_lag = p
    
    def __repr__(self):
        """
        Returns a string representation of the DAR model.
        """
        return f"DAR({self.p}) model assuming {self.dist} errors"
    
    def t(self, Y):
        t = np.arange(start = -1, stop = -(len(Y) + 1), step = -1)
        return t
    
    def y(self, Y):
        Y = np.vstack((np.full((self.max_lag, Y.shape[1]), np.nan), Y))
        y = lambda t: Y[t,:]
        return y
    
    def parse_params(self, flat_params):
        params = self.params
        param_sizes = {key: params[key]["init"].size for key in self.param_names}
        offsets = np.cumsum([0] + list(param_sizes.values())[:-1])
        return {
            key: flat_params[offsets[i]: offsets[i] + size].reshape(params[key]["init"].shape)
            for i, (key, size) in enumerate(param_sizes.items())
            }
    
    def condE(self, Y, params):
        rho = params["rho"]
        y = self.y(Y)
        condE = lambda t: sum(rho[i-1] * y(t-i) for i in range(1, self.p + 1))
        return condE
    
    def condV(self, Y, params):
        omega = params["omega"]
        alpha = params["alpha"]
        y = self.y(Y)
        condV = lambda t: omega + sum(alpha[i-1] * y(t-i)**2 for i in range(1, self.p+1))
        return condV
    
    def fit(self, Y):

        def log_likelihood(params):
            t, y = self.t(Y), self.y(Y)
            condE, condV = self.condE(Y, params), self.condV(Y, params)

            if self.dist == "student-t":
                nu = params["nu"]
                l_t = -1/2 * np.log(condV(t)) + loggamma((nu + 1)/2) - loggamma(nu/2) -((nu + 1)/2) * np.log(1 + (y(t) - condE(t))**2 / (condV(t) * (nu - 2)))
            else:
                l_t = -1/2 * (np.log(condV(t)) + (y(t) - condE(t))**2 / condV(t))
            return -np.nansum(l_t)
                
        tracker = ParamTracker(self.params)
        self.param_names = (log_likelihood(tracker), tracker)[1].get_keys()
        
        results = scipy.optimize.minimize(
            lambda params: log_likelihood(self.parse_params(params)),
            np.concatenate([self.params[key]['init'].ravel() for key in self.param_names]),
            method = "SLSQP",
            bounds = np.concatenate([self.params[key]['bound'] for key in self.param_names]),
            )
        
        self.params = self.parse_params(results.x)
        
    def companion_matrix(self):
        T_sim = self.T_sim
        rho   = self.params['rho'].reshape(-1, 1)
        alpha = self.params['alpha'].reshape(-1, 1)
        p = self.p

        if self.dist == "student-t":
            nu = self.params["nu"]
            eta = student_t.rvs(df = nu, size = (p, T_sim)) * np.sqrt((nu - 2) / nu)
        else:
            eta = gaussian.rvs(loc = 0, scale=1, size=(p, T_sim))
        
        upper_row = (rho + np.sqrt(alpha) * eta).T
        bottom_rows = np.hstack((np.eye(p - 1), np.zeros((p - 1, 1))))
        A = np.array([np.vstack((row, bottom_rows)) for row in upper_row])
        return A
    
    def kron_power(self, A, n):
        A = torch.from_numpy(A).to(dtype = torch.float64)
        
        results = []
        for matrix in A:    
            result = matrix
            for _ in range(n - 1):
                result = torch.kron(result, matrix)
            results.append(result)
        
        kronekcer_power = torch.stack(results)
        return kronekcer_power.cpu().numpy()
    
    def strict_statio(self):
        A = self.companion_matrix()
        T, p, _ = A.shape

        log_sum = 0.0
        cumulative_matrix = np.eye(p)
        for t in range(T):
            cumulative_matrix = np.dot(cumulative_matrix, A[t])
            norm_val = np.linalg.norm(cumulative_matrix, ord = 'fro')
            if norm_val > 0:
                log_sum += np.log(norm_val)
                cumulative_matrix /= norm_val
            else:
                return float('-inf')
        return log_sum / T
    
    def weak_statio(self):
        A = self.companion_matrix()

        A_kron2= self.kron_power(A,2)
        A_kron2_expected = np.average(A_kron2, axis = 0)
        spectral_radius = max(abs(eigvals(A_kron2_expected)))
        return spectral_radius
    
    def statio(self, T_sim = 100000):
        
        self.T_sim = T_sim
        
        stationarity_dict = {
            "strictly": {"stationary": self.strict_statio() < 0, "top Lyapunov exponent": self.strict_statio()},
            "weakly": {"stationary": self.weak_statio() < 1, "spectral radius of the expected Kronecker product of the companion matrix": self.weak_statio()}
            }
        return stationarity_dict

    def std_res(self, Y):        
        t, y = self.t(Y), self.y(Y)
        condE = self.condE(Y, self.params)
        condV = self.condV(Y, self.params)

        z = lambda t: (y(t) - condE(t)) / np.sqrt(condV(t))
        return np.flip(z(t))

    def predict(self, Y):
        t = self.t(Y)
        condE = self.condE(Y, self.params)

        return np.flip(np.append(condE(t+1), np.nan)).reshape(-1,1) 
    
    def predict_CI(self, Y, alpha_level = 0.05):
        t = self.t(Y)
        condE = self.condE(Y, self.params)
        condV = self.condV(Y, self.params)

        if self.dist == "student-t":
            nu = self.params["nu"]
            z = student_t.ppf(1 - alpha_level / 2, df = nu) * np.sqrt((nu - 2) / nu)
        else:
            z = gaussian.ppf(1 - alpha_level / 2)

        CI = lambda t: np.column_stack((condE(t) + z * np.sqrt(condV(t)), 
                                        condE(t) - z * np.sqrt(condV(t))
                                        ))
        
        return np.vstack((np.full((1, 2), np.nan), np.flip(CI(t+1))))