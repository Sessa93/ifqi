import numpy as np
from ifqi import envs
from ifqi.evaluation import evaluation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
import pickle
import time

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def fit_gp(X, X_train, X_test, y_train, y_test, kernel):
      
    start = time.time()
    
    print("Prior: " + str(kernel))
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    print("Fitting GP")
    gp.fit(X_train,y_train)
    print("Posterior: " + str(gp.kernel_))
    
    if np.shape(X_test)[0] > 0:
        print("Testing")
        y_pred = gp.predict(X_test)
    
        score = mean_squared_error(y_test, y_pred)
        print("MSE: " + str(score))
        
    print("Predicting")
    y_pred, std_pred = gp.predict(X, return_std=True)
    
    del gp
    
    end = time.time()
    
    print("Total time: " + str(end - start))
    
    return (y_pred,std_pred)

# Number of source samples (in years)
n_source = 30
# Test fraction
test_fraction = 0.3
# Source ID
source_id = 1
# Whether data should be loaded
load_data = False
# Whether to fit reward
fit_rw = True
# Whether to fit transfition
fit_st = False

# Kernels
kernel_rw = 20.0 * Matern(length_scale=10.0, length_scale_bounds=(1e-1, 100.0), nu=1.5)
kernel_st = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01,1000.0)) + 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)

# Tasks definition
source_mdp_1 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
source_mdp_4 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdps = [source_mdp_1, source_mdp_2, source_mdp_3, source_mdp_4]
mdp = source_mdps[source_id-1]

# Either load or generate episodes
if load_data:
    print("Loading data")
    data = load_object("source_data_" + str(source_id) + ".pkl")
    source_samples = data[0]
    rw_pred = data[1]
    st_pred = data[2]
else:
    print("Collecting episodes")
    source_policy = load_object('source_policy_dam_'+str(source_id)+'.pkl')
    source_samples = evaluation.collect_episodes(mdp, source_policy, n_episodes=n_source)
    rw_pred = None
    st_pred = None
        
X = source_samples[:,0:3]

if fit_rw:
    print("Fitting reward GP")
    y = source_samples[:,3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_fraction)
    rw_pred = fit_gp(X, X_train, X_test, y_train, y_test, kernel_rw)

if fit_st:
    print("Fitting transition GP")
    y = source_samples[:,4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_fraction)
    st_pred = fit_gp(X, X_train, X_test, y_train, y_test, kernel_st)

data = [source_samples, rw_pred, st_pred]
save_object(data, "source_data_" + str(source_id) + ".pkl")