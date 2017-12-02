import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.regressor import Regressor
import json
import scipy.stats as stats
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
# Number of repetitions of each run
n_runs = 10
# Variance of the reward model
var_rw = 0.5
# Variance of the transition model
var_st = 20.0
# Weight discard threshold
max_weight = 1000

# File to save results
perf_file = open('perf_dam_transfer_ideal_' + str(n_source) + '.txt', 'w')

# Tasks definition
source_mdp_1 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 2, inflow_std = 4.0, alpha = 0.8, beta = 0.2)
source_mdp_2 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 3, inflow_std = 4.0, alpha = 0.35, beta = 0.65)
source_mdp_3 = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 4, inflow_std = 4.0, alpha = 0.7, beta = 0.3)
target_mdp = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 4.0, alpha = 0.3, beta = 0.7)

source_mdps = [source_mdp_1, source_mdp_2, source_mdp_3]

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

# Initialize dataset and weights
dataset = []
ws = []
wr = []

if n_source > 0:
    
    for k in range(len(source_mdps)):
        
        # k-th source mdp
        mdp = source_mdps[k]
        
        print("Collecting episodes for source " + str(k))
        source_policy = load_object('source_policy_dam_'+str(k+1)+'.pkl')
        source_samples = evaluation.collect_episodes(mdp, source_policy, n_episodes=n_source)
        
        for i in range(len(source_samples)):
                
            # Get i-th sample
            (s1,d1,a,r,s2,d2,f1,f2) = source_samples[i] # storage,day,action,reward,nextstorage,nextday,end,terminal
            
            mu_rw_t = target_mdp.get_reward(s1,a)
            mu_rw_s = mdp.get_reward(s1,a)
            num_rw = stats.norm.pdf(r, mu_rw_t, math.sqrt(var_rw))
            denom_rw = stats.norm.pdf(r, mu_rw_s, math.sqrt(var_rw))
            w_rw = num_rw / denom_rw
            
            mu_st_t = s1 + target_mdp.INFLOW_MEAN[int(d1-1)] - a
            mu_st_s = s1 + mdp.INFLOW_MEAN[int(d1-1)] - a
            num_st = stats.norm.pdf(s2, mu_st_t, math.sqrt(var_st))
            denom_st = stats.norm.pdf(s2, mu_st_s, math.sqrt(var_st))
            w_st = num_st / denom_st
        
            # Discard very large weights
            if w_rw < max_weight and w_st < max_weight: 
                dataset.append([s1,d1,a,r,s2,d2,f1,f2])
                wr.append(w_rw)
                ws.append(w_st)
        
print("Total source samples: " + str(len(dataset)))

n_target_old = 0
for n_target in [1,5,10,20,30,40,50,100]:
    
    if n_target > 1:
        dataset = dataset.tolist()
        wr = wr.tolist()
        ws = ws.tolist()
    
    print("Starting N = " + str(n_target))
    # List storing the performance of each run
    evals = [n_target]
    
    # Generate target samples
    print("Collecting target episodes")
    target_samples = evaluation.collect_episodes(target_mdp,n_episodes=(n_target - n_target_old))

    n_target_old = n_target
    
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
    
    sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)
    
    # Run WFQI n_runs times
    for e in range(n_runs):

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
