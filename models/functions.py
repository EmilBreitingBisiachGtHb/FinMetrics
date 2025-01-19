import numpy as np
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA # skal implementeres i funktionerne

from scipy.special import gammaln
from scipy.stats import norm, t as student_t


def VARlsExog(y, p, con, tr, exog):
    # Define dependent and independent variables for VAR estimation
    T, K = y.shape
    dep = y[p:T, :]
    
    # Independent variable
    indep = lagmatrix(y, p)
    indep = indep[p:, :]
    
    if con == 1:
        indep = np.hstack([indep, np.ones((len(indep), 1))])
    
    if tr == 1:
        indep = np.hstack([indep, np.arange(1, len(indep) + 1).reshape(-1, 1)])
    
    if isinstance(exog, np.ndarray):
        indep = np.hstack([indep, exog[p:, :]])
    
    T, Kp = indep.shape
    
    # Beta estimation
    Beta = np.linalg.inv(indep.T @ indep) @ indep.T @ dep

    # Residuals
    res = dep - indep @ Beta
    
    # Covariance matrix of residuals
    SIGMA = (res.T @ res) / T
    
    # Covariance matrix of Beta
    CovBeta = np.kron(SIGMA, np.linalg.inv(indep.T @ indep))
    
    tratioBeta = Beta.reshape(-1, order='F') / np.sqrt(np.diag(CovBeta))

    SEbeta = np.sqrt(np.diag(CovBeta)).reshape(-1, order='F')

    df = T - Kp  # degrees of freedom
    Pvalue = 2 * (1 - t.cdf(np.abs(tratioBeta), df))

    m = p
    aiccrit = np.log(np.linalg.det(SIGMA)) + 2 / T * (m * K ** 2 + K)       # AIC value
    hqccrit = np.log(np.linalg.det(SIGMA)) + 2 * np.log(np.log(T)) / T * (m * K ** 2 + K)  # HQC value
    siccrit = np.log(np.linalg.det(SIGMA)) + np.log(T) / T * (m * K ** 2 + K)  # SIC value

    return Beta, SEbeta, CovBeta, Pvalue, tratioBeta, res, indep, SIGMA, aiccrit, hqccrit, siccrit

def lagmatrix(data, lags):
    T, K = data.shape
    lagged_data = np.zeros((T, K * lags))
    for lag in range(1, lags + 1):
        lagged_data[lag:, (lag - 1) * K:lag * K] = data[:T - lag, :]
    return lagged_data


def sdummy(nobs,freq):
    """
    PURPOSE: creates a matrix of seasonal dummy variables
    ---------------------------------------------------
    USAGE: y = sdummy(nobs,freq)
    where: freq = 4 for quarterly, 12 for monthly
    ---------------------------------------------------
    RETURNS:
    y = an (nobs x freq) matrix with 0's and 1's
    e.g., 1 0 0 0 (for freq=4)
    0 1 0 0
    0 0 1 0
    0 0 0 1
    1 0 0 0
    ---------------------------------------------------
    """

    nobs_new = nobs + (freq - (nobs % freq))
    seas = np.zeros([nobs_new,freq])
    
    for i in range(1, nobs_new, freq):
        seas[i-1:i+freq-1,0:freq] = np.identity(freq)
    
    seas = seas[:nobs]
    return seas


def pfind(y, pmax):
    t, K = y.shape

    # Construct regressor matrix and dependent variable
    XMAX = np.ones((1, t - pmax))
    
    for i in range(1, pmax + 1):
        XMAX = np.vstack([XMAX, y[pmax - i:t - i, :].T])

    Y = y[pmax:t, :].T

    aiccrit = np.zeros((pmax + 1, 1))
    hqccrit = np.zeros((pmax + 1, 1))
    siccrit = np.zeros((pmax + 1, 1))

    # Evaluate criterion for p = 0,...,pmax
    for j in range(0, pmax + 1):
        m = j
        T = t - pmax
        X = XMAX[:j * K + 1, :]

        B = Y @ X.T @ np.linalg.inv(X @ X.T)

        SIGMA = (Y - B @ X) @ (Y - B @ X).T / T

        aiccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 / T * (m * K ** 2 + K)       # AIC value
        hqccrit[j] = np.log(np.linalg.det(SIGMA)) + 2 * np.log(np.log(T)) / T * (m * K ** 2 + K)  # HQC value
        siccrit[j] = np.log(np.linalg.det(SIGMA)) + np.log(T) / T * (m * K ** 2 + K)  # SIC value

    # Rank models for p = 0,1,2,...,pmax
    aichat = np.argmin(aiccrit)
    hqchat = np.argmin(hqccrit)
    sichat = np.argmin(siccrit)

    infomat = np.hstack([siccrit, hqccrit, aiccrit])
    m = np.arange(0, pmax + 1).reshape(-1, 1)
    imat = np.hstack([m, infomat])

    LagInformationValue = pd.DataFrame(imat, columns=['Lag', 'SIC', 'HQ', 'AIC'])
    OptimalLag = pd.DataFrame([[sichat, hqchat, aichat]], columns=['SIC', 'HQ', 'AIC'])

    return LagInformationValue, OptimalLag

def out_of_sample_pred(y_test, y_pred):
    y_pred_lagged = y_pred[168::]
    y_test_lagged = y_test[168::]

    # sMAPE Calculation
    numerator = np.abs(y_pred_lagged - y_test_lagged)
    denominator = (np.abs(y_test_lagged) + np.abs(y_pred_lagged)) / 2.0

    epsilon = 1e-32  # A small constant to avoid division by zero
    denominator = np.where(denominator == 0, epsilon, denominator)

    smape_values = numerator / denominator
    smape = np.nanmean(smape_values) * 100


    # RMSE Calculation
    mse = np.nanmean((y_pred_lagged - y_test_lagged) ** 2)
    rmse = np.sqrt(mse)

    # rMAE Calculation
    y_naive = y_test[:-168]

    numerator = np.nanmean(np.abs(y_test_lagged - y_pred_lagged))
    denominator = np.nanmean(np.abs(y_test_lagged - y_naive))

    epsilon = 1e-32
    denominator = max(denominator, epsilon)

    rmae = numerator / denominator * 100

    return rmse, smape, rmae

def smape(y_test, y_pred):
    numerator = np.abs(y_pred - y_test)
    denominator = (np.abs(y_test) + np.abs(y_pred)) / 2.0

    epsilon = 1e-32  # A small constant to avoid division by zero
    denominator = np.where(denominator == 0, epsilon, denominator)

    # Calculate sMAPE
    smape_values = numerator / denominator
    smape = np.mean(smape_values) * 100

    return smape

def VARLMtest(y, p, con, tr, exog, h):
    from scipy.stats import chi2, f

    t, K = y.shape
    Beta, SEbeta, CovBeta, Pvalue, tratioBeta, residuals_u, indep, SIGMA_u, aiccrit, hqccrit, siccrit = VARlsExog(y, p, con, tr, exog)
    u_lags = lagmatrix(residuals_u, h)
    Beta, SEbeta, CovBeta, Pvalue, tratioBeta, residuals_e, indep, SIGMA_e, aiccrit, hqccrit, siccrit = VARlsExog(y, p, con, tr, u_lags)
    LML = (t - p)*(K-np.trace(np.linalg.inv(SIGMA_u) @ SIGMA_e));
    LMLpval = 1-chi2.cdf(LML,h*K**2);

    m = K * h
    q = 1/2 * K * m - 1
    s = ((K**2 * m**2 - 4) / (K**2 + m**2 - 5))**(1/2)
    N = (t - p) - K * p - m - (1/2) * (K - m + 1)

    FLMh = ((np.linalg.det(SIGMA_u) / np.linalg.det(SIGMA_e))**(1/s) - 1) * ((N * s - q) / (K * m))
    FLMh_pval = 1 - f.cdf(FLMh, h * K**2, N * s - q)

    
    Results = [[LML, FLMh], [LMLpval, FLMh_pval], [h, h]]

    lm_table = pd.DataFrame({
        'Measure': pd.Categorical(['Test statistic', 'p-value', 'Lag order']),
        'Breusch_Godfrey': [row[0] for row in Results],
        'Edgerton_Shukur': [row[1] for row in Results]
    })

    return Results, lm_table


def plot_actual_vs_predicted(y_test, predictions, graph_name):
    plt.figure(figsize=(8, 4))
    plt.plot(y_test[-168:], label='Actual', color='blue', alpha=1, linewidth=0.5)
    plt.plot(predictions[-168:], label='Predicted', color='red', alpha=1, linewidth=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(graph_name, fontsize=14)
    
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.show()

def combine_24_hour_data(predictions_dict, y_test):
    n_days = min(len(predictions_dict[hour]) for hour in range(24))

    combined_predictions = []

    for day in range(n_days):
        for hour in range(24):
            combined_predictions.append(predictions_dict[hour][day])

    combined_predictions = np.array(combined_predictions).reshape(-1, 1)

    combined_test = []
        
    for day in range(n_days):
        for hour in range(24):
            combined_test.append(y_test[hour][day])

    combined_test = np.array(combined_test).reshape(-1, 1)

    return combined_predictions, combined_test


def PCA_dimreduc(training_data, test_data, exp_var_pct):
    from sklearn.decomposition import PCA

    # Compute mean (bias) of training data
    PCA_bias = np.mean(training_data, axis=0)

    # Find number of loadings mathcing the level of explained variance
    pca = PCA()
    pca.fit(training_data)
    components = pca.components_
    eigvals_i = (pca.singular_values_) ** 2 # eigenvalues equals the squared singular values
    explained_variance_cumsum = np.array(list(enumerate((np.cumsum(eigvals_i) / np.sum(eigvals_i)), start=1))) # computing cummulative explained variancec with correct index
    K_loadings = min(np.where(explained_variance_cumsum[:,1] >= exp_var_pct)[0]) + 1 # number of components to explain X pct. variance

    # Extract first K principal components (PCA loadings) necessary to match the chosen level of explained variance
    PCA_loadings = components[0:K_loadings, :].T

    # Compute dimensinality reduced data by projecting the centered test_data onto the first K principal components
    data_dimreduc = (test_data - PCA_bias) @ PCA_loadings

    return data_dimreduc, PCA_loadings, PCA_bias, K_loadings, explained_variance_cumsum


def deseasonalize(df, season_pattern):
    import holidays
    from sklearn.linear_model import LinearRegression
    df = df.copy()

    df['DATE'] = pd.to_datetime(df['DATE'])

    df['weekday'] = df['DATE'].dt.weekday  # 0=Monday, 6=Sunday
    df['Omega_t'] = df.groupby('weekday')['DK2_spot'].transform('mean')

    df['is_weekend'] = df['DATE'].dt.weekday >= 5  # True for Saturday (5) and Sunday (6)

    denmark_holidays = holidays.Denmark()

    df['is_holiday'] = df['DATE'].apply(lambda x: x in denmark_holidays)

    df['D_t'] = np.where(df['is_weekend'] | df['is_holiday'], 1, 0)

    df['psi'] = 1

    for i in range(1, 13):
        df[f'M_{i}'] = np.where(df['DATE'].dt.month == i, 1, 0)

    predictors = ['psi'] + ['D_t'] + [f'M_{i}' for i in range(2, 13)]  # D_t and M_2 to M_12

    train_data = df[df['DATE'] < "2023-01-01"]
    predict_data = df[df['DATE'] >= "2023-01-01"]

    X_train = train_data[predictors]
    y_train = train_data['DK2_spot']

    model = LinearRegression()
    model.fit(X_train, y_train)

    df['Xi_t'] = model.predict(df[predictors])

    if season_pattern == "short":
        df['deseasonalized'] = df['DK2_spot'] - df['Omega_t']
        df['seasonal_component'] = df['Omega_t']

    elif season_pattern == "long":
        df['deseasonalized'] = df['DK2_spot'] - df['Xi_t']
        df['seasonal_component'] = df['Xi_t']

    elif season_pattern == "both":
        df['deseasonalized'] = df['DK2_spot'] - df['Omega_t'] - df['Xi_t']
        df['seasonal_component'] = df['Omega_t'] + df['Xi_t']
    
    return df


def split_data_into_series(datasets, pca_percent, regex_choice):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    datasets_train = {hour: datasets[hour][(datasets[hour]['DATE'] >= '2014-01-01') & (datasets[hour]['DATE'] < '2023-01-01')] for hour in range(24)}
    datasets_test = {hour: datasets[hour][datasets[hour]['DATE'] >= '2023-01-01'] for hour in range(24)}

    y_train = {hour: datasets_train[hour]['DK2_spot'].to_numpy().reshape(-1, 1) for hour in range(24)}
    y_test = {hour: datasets_test[hour]['DK2_spot'].to_numpy().reshape(-1, 1) for hour in range(24)}

    y_train_deseason = {hour: datasets_train[hour]['deseasonalized'].to_numpy().reshape(-1, 1) for hour in range(24)}
    y_test_deseason = {hour: datasets_test[hour]['deseasonalized'].to_numpy().reshape(-1, 1) for hour in range(24)}

    y_train_season = {hour: datasets_train[hour]['seasonal_component'].to_numpy().reshape(-1, 1) for hour in range(24)}
    y_test_season = {hour: datasets_test[hour]['seasonal_component'].to_numpy().reshape(-1, 1) for hour in range(24)}

    exog_variables_train = {
        hour: datasets_train[hour].filter(regex=regex_choice).to_numpy()
        for hour in range(24)
    }

    scalers = {}
    exog_variables_train_stand = {}

    # Fit and transform each hour’s data
    for hour in range(24):
        # Fit the scaler on the training data for the current hour
        scaler = StandardScaler().fit(exog_variables_train[hour])
        scalers[hour] = scaler  # Store the scaler for each hour
        
        # Transform the training data
        exog_variables_train_stand[hour] = scaler.transform(exog_variables_train[hour])

    exog_variables_test = {
        hour: datasets_test[hour].filter(regex=regex_choice).to_numpy()
        for hour in range(24)
    }

    exog_variables_test_stand = {}

    # Transform each hour’s test data using the corresponding scaler
    for hour in range(24):
        # Use the scaler fitted on the training data for this hour
        exog_variables_test_stand[hour] = scalers[hour].transform(exog_variables_test[hour])
        

    pca_train = {}
    pca_test = {}

    for hour in range(24):
        data_dimreduc, _, _, _, _ = PCA_dimreduc(exog_variables_train_stand[hour], exog_variables_train_stand[hour], pca_percent)
        pca_train[hour] = data_dimreduc

        data_dimreduc, _, _, _, _ = PCA_dimreduc(exog_variables_train_stand[hour], exog_variables_test_stand[hour], pca_percent)
        pca_test[hour] = data_dimreduc

    return y_train, y_test, y_train_deseason, y_test_deseason, y_train_season, \
        y_test_season, exog_variables_train, scalers, exog_variables_train_stand, \
            exog_variables_test, exog_variables_test_stand, pca_train, pca_test


def prepend_nan(data):

    return np.vstack([
            np.full((1, data.shape[1]), np.nan), 
            data
        ])



#######################################################################################################################################
# DAR
#######################################################################################################################################

# DAR class

class DAR:
    def __init__(self, p):
        """
        Initialize the DAR class
        """
        self.p = p
        self.params = None
        self.num_params = 1 + 2 * self.p
        self.loglike_val = None

    def t(self, Y):
        T = len(Y)
        self.T = T
        t = np.arange(start=-1, stop=-(T+1), step=-1)
        
        return t
    
    def y(self, Y):
        
        max_lag = self.p 

        Y = np.vstack((np.full((max_lag, Y.shape[1]), np.nan), Y))
        y = lambda t: Y[t,:]
        
        return y

    def parse_params(self, params):
        rho = params[:self.p]
        omega = params[self.p]
        alpha = params[self.p + 1:self.p + 1 + self.p]
        
        params_dict = {
            "rho": rho,
            "omega": omega,
            "alpha": alpha,
            }
        return params_dict

    def cond_var(self, Y, params = None):
        
        omega = params["omega"]
        alpha = params["alpha"]

        y = self.y(Y)
        
        cond_var = lambda t: omega + sum(alpha[i-1] * y(t-i)**2 for i in range(1, self.p+1))
        
        return cond_var
    
    def cond_mean(self, Y, params = None):
        
        rho = params["rho"]
        
        y = self.y(Y)
        cond_mean = lambda t: sum(rho[i-1] * y(t-i) for i in range(1, self.p + 1))
        return cond_mean
    
    def fit(self, Y, dist = "normal"):

        t = self.t(Y)

        def MLE(params):
            
            y = self.y(Y)

            if dist == "student-t":
                nu = params[-1]
                params = self.parse_params(params[:-1])
            
            else:
                params = self.parse_params(params)
            
            cond_mean = self.cond_mean(Y, params)
            cond_var = self.cond_var(Y, params)

            if dist == "normal":
                l_t = -1/2 * (np.log(cond_var(t)) + (y(t) - cond_mean(t))**2 / cond_var(t))
            
            elif dist == "student-t":
                term1 = -0.5 * np.log(cond_var(t) * (nu - 2) * np.pi)
                term2 = gammaln((nu + 1) / 2) - gammaln(nu / 2)
                term3 = -((nu + 1) / 2) * np.log(1 + ((y(t) - cond_mean(t)) ** 2) / (cond_var(t) * (nu - 2)))
                l_t = term1 + term2 + term3

            else:
                raise ValueError("dist must be 'normal' or 'student-t'")
            
            L_t = np.nansum(l_t)
            return -L_t
        
        # Initialize parameters
        rho_init = [0.0] * self.p
        omega_init = [0.05]
        alpha_init = [0.5] * self.p
        init_params = rho_init + omega_init + alpha_init

        # Set parameter bounds
        rho_bound = [(None, None)] * self.p
        omega_bound = [(0.001, None)]
        alpha_bound = [(0.001, None)] * self.p
        bounds_params = rho_bound + omega_bound + alpha_bound

        # studen-t degrees of freedom
        nu_init = [5]
        nu_bound = [(2.0001, None)]
        
        if dist == "student-t":
            init_params += nu_init
            bounds_params += nu_bound
        
        # opminise the loglikelihood function
        result = scipy.optimize.minimize(MLE, init_params, method='SLSQP', bounds=bounds_params)
        
        # store parameters
        if dist == "student-t":
            self.params = {**self.parse_params(result.x[:-1]), "nu": result.x[-1]}
        else:
            self.params = self.parse_params(result.x)

        # store loglikelihood value
        self.loglike_val = -result.fun
        
        return self

    def std_res(self, Y):
        
        t = self.t(Y)
        y = self.y(Y)

        cond_mean = self.cond_mean(Y, self.params)
        cond_var = self.cond_var(Y, self.params)

        z = lambda t: (y(t) - cond_mean(t)) / np.sqrt(cond_var(t))
        return np.flip(z(t))

    def predict(self, Y):
        
        t = self.t(Y)
        
        cond_mean = self.cond_mean(Y, self.params)
        
        return np.flip(np.append(cond_mean(t+1), np.nan)).reshape(-1,1)
    
    def CI(self, Y, alpha_level = 0.05, dist = "normal"):

        t = self.t(Y)
        cond_mean = self.cond_mean(Y, self.params)
        cond_var = self.cond_var(Y, self.params)

        if dist == "normal":
            z_val = norm.ppf(1 - alpha_level / 2)

        elif dist == "student-t":
            nu = self.params["nu"]
            z_val = student_t.ppf(1 - alpha_level / 2, df = nu) * np.sqrt((nu - 2) / nu)
        
        else:
            raise ValueError("dist must be 'normal' or 'student-t'")
        

        CI = lambda t: np.column_stack((cond_mean(t) + z_val * np.sqrt(cond_var(t)), 
                                        cond_mean(t) - z_val * np.sqrt(cond_var(t))
                                        ))
        
        return np.vstack((np.full((1, 2), np.nan), np.flip(CI(t+1))))


# Simulate DAR

def DAR_sim(parameters, T, set_seed):
    np.random.seed(set_seed)
    rho, omega, alpha = parameters

    max_lag = 1
    
    # Initialize arrays
    Y = np.zeros(T)
    Z = np.random.normal(loc=0, scale=1, size=T)

    # Initialize the process
    Y[0] = 1 

    # Define function objects at time t
    y = lambda t: Y[t]
    z = lambda t: Z[t]
    sigma = lambda t: np.sqrt(omega + alpha * Y[t-1]**2)
    epsilon = lambda t: sigma(t) * z(t)

    # The model
    DAR = lambda t: rho * y(t-1) + epsilon(t)

    # Update Y
    for t in range(max_lag, T):
        Y[t] = DAR(t)
    return Y


#######################################################################################################################################
# DARMA-X
#######################################################################################################################################

# DARMA-X class

class DARMAX:
    def __init__(self, p):
        """
        Initialize the DARMA-X class
        """
        self.p = p
        self.d = 0
        self.num_params = None
        self.params = None
        self.loglike_val = None
        self.T = None
    
    def t(self, Y):
        T = len(Y)
        self.T = T
        t = np.arange(start=-1, stop=-(T+1), step=-1)
        
        return t
    
    def y(self, Y):
        
        max_lag = self.p + 1

        Y = np.vstack((np.full((max_lag, Y.shape[1]), np.nan), Y))
        y = lambda t: Y[t,:]
        
        return y
    
    def x(self, X = None):

        max_lag = self.p + 1

        if X is None:
            x = lambda t: np.zeros((self.d,))
        else:
            X = np.vstack((np.full((max_lag, X.shape[1]), np.nan), X))
            x = lambda t: X[t, :]
        
        return x

    def parse_params(self, params):
        p, d = self.p, self.d

        rho = params[:p]
        omega, phi = params[p], params[p + 1]
        alpha = params[p + 2:p + 2 + p]
        gamma = params[p + 2 + p:p + 2 + p + p * d].reshape(p, d) if p * d > 0 else np.zeros((p, d))
        nu = params[-1] if len(params) > p + 2 + p + p * d else None

        return {
            "rho": rho,
            "omega": omega,
            "phi": phi,
            "alpha": alpha,
            "gamma": gamma,
            **({"nu": nu} if nu is not None else {})
        }

    def cond_var(self, Y, X = None, params = None):
        
        omega = params["omega"]
        alpha = params["alpha"]
        gamma = params["gamma"]

        y = self.y(Y)
        x = self.x(X)

        sigma = lambda t: np.sqrt(omega
                                  + sum(alpha[i-1] * y(t-i)**2 for i in range(1, self.p+1))
                                  + sum((gamma[i-1] @ x(t-i).T**2).reshape(-1, 1) for i in range(1, self.p+1))
                                  )
        
        cond_var = lambda t: sigma(t)**2

        return cond_var
    
    def cond_mean(self, Y, X = None, params = None):
  
        rho = params["rho"]
        omega = params["omega"]
        alpha = params["alpha"]
        gamma = params["gamma"]
        phi = params["phi"]

        y = self.y(Y)
        x = self.x(X)
        
        sigma = lambda t: np.sqrt(omega
                                  + sum(alpha[i-1] * y(t-i)**2 for i in range(1, self.p+1))
                                  + sum((gamma[i-1] @ x(t-i).T**2).reshape(-1, 1) for i in range(1, self.p+1))
                                  )
        
        beta = lambda t: phi * (y(t) - sum(rho[i-1] * y(t-i) for i in range(1, self.p + 1))) / sigma(t)

        cond_mean = lambda t: sum(rho[i-1] * y(t-i) for i in range(1, self.p + 1)) + sigma(t) * beta(t-1)

        return cond_mean
    
    def fit(self, Y, X = None, dist = "normal"):

        t = self.t(Y)
        self.d = 0 if X is None else X.shape[1]
        self.num_params = 2 + (2 + self.d) * self.p + (1 if dist == "student-t" else 0)
        
        def MLE(params):
            
            params = self.parse_params(params)
            
            y = self.y(Y)
            cond_var = self.cond_var(Y, X, params)
            cond_mean = self.cond_mean(Y, X, params)

            if dist == "normal":
                l_t = -1/2 * (np.log(cond_var(t)) + (y(t) - cond_mean(t))**2 / cond_var(t))
            
            elif dist == "student-t":
                nu = params["nu"]  # Degrees of freedom
                term1 = -0.5 * np.log(cond_var(t) * (nu - 2) * np.pi)
                term2 = gammaln((nu + 1) / 2) - gammaln(nu / 2)
                term3 = -((nu + 1) / 2) * np.log(1 + ((y(t) - cond_mean(t))**2) / (cond_var(t) * (nu - 2)))
                l_t = term1 + term2 + term3

            else:
                raise ValueError("dist must be 'normal' or 'student-t'")
            
            L_t = np.nansum(l_t)
            return -L_t
        
        init_params = np.full(self.num_params, 0.5)
        if dist == "student-t":
            init_params[-1] = 3  # Set nu to 3 explicitly
        
        bounds = ([(None, None)] * self.p  # rho
                + [(1e-6, None)]  # omega
                + [(-1 + 1e-6, 1 - 1e-6)]  # phi
                + [(1e-6, None)] * self.p  # alpha
                + [(1e-6, None)] * (self.p * self.d)  # gamma
                + ([(2 + 1e-6, None)] if dist == "student-t" else [])  # nu (degrees of freedom) if student-t
                )
        
        result = scipy.optimize.minimize(MLE, init_params, method = "SLSQP", bounds = bounds)
        self.params = self.parse_params(result.x)
        self.loglike_val = -result.fun
        
        return self
    
    def std_res(self, Y, X = None):
        
        t = self.t(Y)
        y = self.y(Y)

        cond_mean = self.cond_mean(Y, X, self.params)
        cond_var = self.cond_var(Y, X, self.params)

        z = lambda t: (y(t) - cond_mean(t)) / np.sqrt(cond_var(t))
        return np.flip(z(t))

    def predict(self, Y, X = None):
        
        t = self.t(Y)
        
        cond_mean = self.cond_mean(Y, X, self.params)
        
        return np.flip(np.append(cond_mean(t+1), np.nan)).reshape(-1,1)
        
    def CI(self, Y, X = None, alpha_level = 0.05, dist = "normal"):

        t = self.t(Y)
        cond_mean = self.cond_mean(Y, X, self.params)
        cond_var = self.cond_var(Y, X, self.params)

        if dist == "normal":
            z_val = norm.ppf(1 - alpha_level / 2)

        elif dist == "student-t":
            nu = self.params["nu"]
            z_val = student_t.ppf(1 - alpha_level / 2, df = nu) * np.sqrt((nu - 2) / nu)
        
        else:
            raise ValueError("dist must be 'normal' or 'student-t'")
        

        CI = lambda t: np.column_stack((cond_mean(t) + z_val * np.sqrt(cond_var(t)), 
                                        cond_mean(t) - z_val * np.sqrt(cond_var(t))
                                        ))
        
        return np.vstack((np.full((1, 2), np.nan), np.flip(CI(t+1))))

# Simulate DARMA

def DARMA_sim(parameters, T, set_seed):
    np.random.seed(set_seed)
    rho, omega, alpha, phi = parameters

    max_lag = 2
    
    # Initialize arrays
    Y = np.zeros(T)
    ETA = np.random.normal(loc=0, scale=1, size = T)

    # Initialize the process
    Y[0] = 1 

    # Define function objects at time t
    y = lambda t: Y[t]
    eta = lambda t: ETA[t]
    sigma = lambda t: np.sqrt(omega + alpha * y(t-1)**2)
    beta = lambda t: phi * (y(t) - rho * y(t-1)) / sigma(t)
    
    # The model
    DARMA = lambda t: rho * y(t-1) + sigma(t) * beta(t-1) + sigma(t) * eta(t)

    # Recursively comput the model
    for t in range(max_lag, T):
        Y[t] = DARMA(t)
    return Y


#######################################################################################################################################
# DimReduc (plot PCA)
#######################################################################################################################################

class DimReduc:
    def __init__(self, pct):

        self.pct = pct
        self.k = None
        self.Z = None
        self.plot_var = None
    
    def fit(self, X):
        pca = PCA().fit(X)
        eigvals_i = (pca.singular_values_) ** 2
        explained_var = np.array(list(enumerate((np.cumsum(eigvals_i) / np.sum(eigvals_i)), start=1)))
        
        
        k = min(np.where(explained_var[:,1] >= self.pct)[0]) + 1
        self.k = k
        self.plot_var = explained_var

def plot_explained_variance(X, pct):
    fig, axs = plt.subplots(4, 6, figsize=(20, 12))
    axs = axs.flatten()
    for i in range(24):
        model = DimReduc(pct=pct)
        model.fit(X[i])
        plot = model.plot_var
        
        ax = axs[i]
        ax.plot(plot[:, 0], plot[:, 1], label='Explained Variance', marker='o')
        ax.axvline(x=model.k, color='navy', linestyle='--', label=f'Top $k = {model.k}$ Principal Components')
        ax.legend(prop={'size': 9})
        ax.grid(True)
        ax.set_title(f'Hour {i}')
    
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()

#plot_explained_variance(X, pct = 0.85)