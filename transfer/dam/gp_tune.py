import sys
import time
import numpy as np
from ifqi import envs
from ifqi.evaluation import evaluation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

n_episodes = 30

source_mdp_1 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
source_mdp_4 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
target_mdp = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)
mdps = [source_mdp_1, source_mdp_2, source_mdp_3, source_mdp_4, target_mdp]
mdp = mdps[0]

samples = evaluation.collect_episodes(mdp, n_episodes=n_episodes) # storage,day,action,reward,nextstorage,nextday,end,terminal

X = samples[:,0:3]
#y = samples[:,3]
y = samples[:,4]
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scale_bounds = (0.01,1000.0)

kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=scale_bounds),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * Matern(length_scale=1.0, length_scale_bounds=scale_bounds, nu=1.5),
           1.0 * RBF(length_scale=1.0, length_scale_bounds=scale_bounds) + 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
           1.0 * RBF(length_scale=1.0, length_scale_bounds=scale_bounds) + 1.0 * Matern(length_scale=1.0, length_scale_bounds=scale_bounds, nu=1.5),
           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1) + 1.0 * Matern(length_scale=1.0, length_scale_bounds=scale_bounds, nu=1.5),
           1.0 * RBF(length_scale=1.0, length_scale_bounds=scale_bounds) + 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1) + 
           1.0 * Matern(length_scale=1.0, length_scale_bounds=scale_bounds, nu=1.5)]

for kernel in kernels:
    
    start = time.time()
    
    print("--------------------------------")
    print("Prior: " + str(kernel))
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    print("Fitting GP")
    gp.fit(X_train,y_train)
    print("Predicting")
    y_pred, std_pred = gp.predict(X_test,return_std=True)
    
    print("Posterior: " + str(gp.kernel_))

    score = mean_squared_error(y_test, y_pred)
    print("MSE: " + str(score))
        
    end = time.time()
    
    print("Total time: " + str(end - start))
    
sys.exit()

values0 = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
values1 = np.array([1.0, 10.0, 50.0, 100.0, 500.0])
values2 = np.array([0.001, 0.01, 0.1, 1.0])

values = np.array(np.meshgrid(values0, values1, values2)).T.reshape(-1,3).tolist()

best = float("Inf")

for value in values:

    start = time.time()
    
    print("--------------------------------")
    
    kernel = value[0] * RationalQuadratic(length_scale = value[1], alpha = value[2])
    print(kernel)
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10)
    print("Fitting GP")
    gp.fit(X_train,y_train)
    print("Predicting")
    y_pred, std_pred = gp.predict(X_test,return_std=True)
    
    print(gp.kernel_)

    score = mean_squared_error(y_test, y_pred)
    print(score)
    
    if score < best:
        best = score
        best_params = value
        
    end = time.time()
    
    print("Total time: " + str(end - start))
 
print("--------------------------------")       
print("Best params " + str(best_params) + " with score " + str(best))