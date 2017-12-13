import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.regressor import Regressor
import json
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic, WhiteKernel)
import pickle
import json
import math

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

# Number of source samples (in years)
n_source = 30
# Maximum number of samples for the GPs (in years)
max_gp = 30
# Number of repetitions of each run
n_runs = 10
# Variance of the reward model
var_rw = 0.5
# Variance of the transition model
var_st = 20.0
# Weight discard threshold
max_weight = 1000

# Kernels
kernel_rw = 20.0 * Matern(length_scale=10.0, length_scale_bounds=(1e-1, 1000.0), nu=1.5)
kernel_st = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.01,1000.0)) + WhiteKernel(noise_level = 10.0, noise_level_bounds=(1.0, 50.0))

# File to save results
perf_file = open('perf_dam_transfer_' + str(n_source) + '.txt', 'w')

# Tasks definition
source_mdp_1 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
source_mdp_4 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
target_mdp = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)

source_mdps = [source_mdp_1, source_mdp_2, source_mdp_3, source_mdp_4]

state_dim, action_dim, reward_dim = envs.get_space_info(source_mdp_1)
assert reward_dim == 1

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':20,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = source_mdp_1.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

if n_source > 0:
    
    # List containing the source reward predictions
    source_predictions_rw = []
    # List containing the source transition predictions
    source_predictions_st = []
    # List containing the source state-action couples
    source_X = []
    # List containing the source samples
    source_samples_list = []
    
    for k in range(len(source_mdps)):
        
        data = load_object("source_data_" + str(k+1) + ".pkl")
        source_samples = data[0]
        source_samples_list.append(source_samples)
        source_X.append(source_samples[:,0:3])
        source_predictions_rw.append(data[1])
        source_predictions_st.append(data[2])
        del source_samples
        del data

for n_target in [1,5,10,20,30,40,50,100]:
    
    print("Starting N = " + str(n_target))
    # List storing the performance of each run
    evals = [n_target]
    
    # Generate target samples
    print("Collecting target episodes")
    if n_target == 1:
        target_samples = evaluation.collect_episodes(target_mdp,n_episodes=n_target)
    else:
        target_samples = np.concatenate((target_samples,evaluation.collect_episodes(target_mdp,n_episodes=(n_target - n_target_old))))

    sast, r = split_data_for_fqi(target_samples, state_dim, action_dim, reward_dim)
    n_target_old = n_target
    
    # Initialize dataset and weights
    dataset = []
    ws = []
    wr = []
    
    if n_source > 0:
        
        # Effective number of samples used to fit the target GP
        n_gp = min(n_target,max_gp)
        
        X_train = sast[(n_target*360-n_gp*360):n_target*360,0:3]

        print("Fitting target reward GP")
        y = r[(n_target*360-n_gp*360):n_target*360]
        gp_target_rw = GaussianProcessRegressor(kernel = kernel_rw, n_restarts_optimizer = 10)
        gp_target_rw.fit(X_train,y)
        
        print("Fitting target transition GP")
        y = sast[(n_target*360-n_gp*360):n_target*360,3]
        gp_target_st = GaussianProcessRegressor(kernel = kernel_st, n_restarts_optimizer = 10)
        gp_target_st.fit(X_train,y)
        
        for k in range(len(source_mdps)):
            
            mdp = source_mdps[k]
            
            print("Predicting source " + str(k))
            X_predict = source_X[k]
            mu_gp_t_rw, std_gp_t_rw = gp_target_rw.predict(X_predict,return_std=True)
            mu_gp_t_st, std_gp_t_st = gp_target_st.predict(X_predict,return_std=True)
            
            # Get source predictions
            mu_gp_s_rw, std_gp_s_rw = source_predictions_rw[k]
            mu_gp_s_st, std_gp_s_st = source_predictions_st[k]
            
            print("Computing weights for source " + str(k))
            source_samples = source_samples_list[k]
            for i in range(len(source_samples)):
                
                # Get i-th sample
                (s1,d1,a,r,s2,d2,f1,f2) = source_samples[i] # storage,day,action,reward,nextstorage,nextday,end,terminal
                
                # Compute variances of the densities in the weight expectation
                var_num_rw = var_rw + math.pow(std_gp_t_rw[i],2)
                var_denom_rw = var_rw - math.pow(std_gp_s_rw[i],2)
                
                var_num_st = var_st + math.pow(std_gp_t_st[i],2)
                var_denom_st = var_st - math.pow(std_gp_s_st[i],2)
                
                # Discard illegal samples
                if var_denom_rw > 0 and var_denom_st > 0:
                    
                    num_rw = stats.norm.pdf(r, mu_gp_t_rw[i], math.sqrt(var_num_rw))
                    denom_rw = stats.norm.pdf(r, mu_gp_s_rw[i], math.sqrt(var_denom_rw))
                    w_rw = (num_rw/denom_rw)*(var_rw/var_denom_rw)
                    
                    num_st = stats.norm.pdf(s2, mu_gp_t_st[i], math.sqrt(var_num_st))
                    denom_st = stats.norm.pdf(s2, mu_gp_s_st[i], math.sqrt(var_denom_st))
                    w_st = (num_st/denom_st)*(var_st/var_denom_st)
                
                    # Clip very large weights
                    w_rw = min(w_rw, max_weight)
                    w_st = min(w_st, max_weight)

                    dataset.append([s1,d1,a,r,s2,d2,f1,f2])
                    wr.append(w_rw)
                    ws.append(w_st)
                else:
                    print("WARNING: discarding sample due to imprecise GP")
            
        del gp_target_rw
        del gp_target_st
    
    print("Total source samples: " + str(len(dataset)))
    print("Total target samples: " + str(len(target_samples)))
    # Add target samples
    for (s1,d1,a,r,s2,d2,f1,f2) in target_samples:
        dataset.append([s1,d1,a,r,s2,d2,f1,f2])
        ws.append(1.0)
        wr.append(1.0)
        
    dataset = np.array(dataset)
    wr = np.array(wr)
    ws = np.array(ws)
    
    N = np.shape(wr)[0]
    wr_mean = np.mean(wr)
    ws_mean = np.mean(ws)
    wr_mean2 = np.mean(np.multiply(wr,wr))
    ws_mean2 = np.mean(np.multiply(ws,ws))
    print("Mean wr: "+str(wr_mean))
    print("Mean ws: "+str(ws_mean))
    print("Eff wr: "+str(N * wr_mean ** 2 / wr_mean2))
    print("Eff ws: "+str(N * ws_mean ** 2 / ws_mean2))
    
    # Run WFQI n_runs times
    for e in range(n_runs):
        
        sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

        fit_params = {'sample_weight': wr}
        
        fqi_iterations = target_mdp.horizon
        fqi = FQI(estimator=regressor,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  discrete_actions=discrete_actions,
                  gamma=target_mdp.gamma,
                  horizon=fqi_iterations,
                  verbose=False)

        # Fit reward with weights wr
        sa,est = fqi.partial_fit(sast, r,**fit_params)
        # Predict using fitted reward
        Q1 = est.predict(sa)

        fit_params = {'sample_weight': ws}
        
        # Run FQI with weights ws
        iterations = 60
        best_j = -float("Inf")
        for i in range(iterations - 1):
            fqi.partial_fit(None, Q1,**fit_params)
            values = evaluation.evaluate_policy(target_mdp, fqi, n_episodes = 1, initial_states = np.array([[100.0,1]]))
            if values[0] > best_j:
                best_j = values[0]
        evals.append(best_j)
        print(str(best_j))
        
    print(list(evals))
    json.dump(evals,perf_file)
