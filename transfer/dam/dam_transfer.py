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
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pickle
import json

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


perf_file = open('perf_dam_transfer_50.txt', 'w')

# Tasks definition
source_mdp_1 = envs.Dam(capacity = 450.0, demand = 9.0, flooding = 250.0, inflow_profile = 2, inflow_std = 1.9, alpha = 0.4, beta = 0.6)
source_mdp_2 = envs.Dam(capacity = 600.0, demand = 11.0, flooding = 200.0, inflow_profile = 3, inflow_std = 2.0, alpha = 0.6, beta = 0.4)
target_mdp = envs.Dam(capacity = 500.0, demand = 10.0, flooding = 200.0, inflow_profile = 1, inflow_std = 1.8, alpha = 0.3, beta = 0.7)

source_mdps = [source_mdp_1, source_mdp_2]

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

n_source = 50
for n_target in [1,5,10,20,30,40,50,100]:
    
    print("Starting N = " + str(n_target))
    
    evals = [n_target]
    
    for e in range(20):
        ws = []
        wr = []
        err = 0
        print('Test no. '+str(e+1))
        
        # SAMPLES LABELING -----------------------
        target_samples = evaluation.collect_episodes(target_mdp,n_episodes=n_target)
        sast, r = split_data_for_fqi(target_samples, state_dim, action_dim, reward_dim)

        dataset = []
        if n_source > 0:
            X = sast[:,0:3]

            y = r
            gp_target_rw = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_target_rw.fit(X,y)
            print("Target task reward GP fitted!")
            
            y = sast[:,3]
            gp_target_st = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_target_st.fit(X,y)
            print("Target task transition GP fitted!")
            
            k = 1

            for mdp in source_mdps:

                #best_policy_source = load_object('source_policy_dam'+str(k)+'.pkl')
                source_samples = evaluation.collect_episodes(mdp, n_episodes=n_source)
                sast, r = split_data_for_fqi(source_samples, state_dim, action_dim, reward_dim)

                X = sast[:,0:3]

                y = r
                gp_source_rw = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_source_rw.fit(X,y)
                print("Source task "+str(k)+" reward GP fitted!")
                
                y = sast[:,3]
                gp_source_st = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_source_st.fit(X,y)
                print("Source task "+str(k)+" transition GP fitted!")

                mu_gp_t_rw, std_gp_t_rw = gp_target_rw.predict(X,return_std=True)
                mu_gp_s_rw, std_gp_s_rw = gp_source_rw.predict(X,return_std=True)
                
                mu_gp_t_st, std_gp_t_st = gp_target_st.predict(X,return_std=True)
                mu_gp_s_st, std_gp_s_st = gp_source_st.predict(X,return_std=True)

                raw_dataset = []
                for i in range(len(source_samples)):
                    (s1,d1,a,r,s2,d2,f1,f2) = source_samples[i] # storage,day,action,reward,nextstorage,nextday,end,terminal
                    
                    sigma = 0.1
                    
                    var_num_rw = sigma + math.pow(std_gp_t_rw[i],2)
                    var_denom_rw = sigma - math.pow(std_gp_s_rw[i],2)
                    
                    var_num_st = sigma + math.pow(std_gp_t_st[i],2)
                    var_denom_st = sigma - math.pow(std_gp_s_st[i],2)
                    
                    if var_denom_rw > 0 and var_denom_st > 0:
                        
                        num_rw = stats.norm.pdf(r, mu_gp_t_rw[i], math.sqrt(var_num_rw))
                        denom_rw = stats.norm.pdf(r, mu_gp_s_rw[i], math.sqrt(var_denom_rw))
                        w_rw = (num_rw/denom_rw)*(sigma/var_denom_rw)
                        
                        num_st = stats.norm.pdf(s2, mu_gp_t_st[i], math.sqrt(var_num_st))
                        denom_st = stats.norm.pdf(s2, mu_gp_s_st[i], math.sqrt(var_denom_st))
                        w_st = (num_st/denom_st)*(sigma/var_denom_st)
                    
                        raw_dataset.append([s1,d1,a,r,s2,d2,f1,f2,w_rw,w_st])
                    else:
                        err += 1
                
                # Weight filtering
                for (s1,d1,a,r,s2,d2,f1,f2,w_rw,w_st) in raw_dataset:
                    if w_rw < 10 and w_st < 10:
                        dataset.append([s1,d1,a,r,s2,d2,f1,f2])
                        wr.append(w_rw)
                        ws.append(w_st)
                del raw_dataset
                k += 1

        print("Total Source Samples: "+str(len(dataset)))
        for (s1,d1,a,r,s2,d2,f1,f2) in target_samples:
            dataset.append([s1,d1,a,r,s2,d2,f1,f2])
            ws.append(1.0)
            wr.append(1.0)
        dataset = np.array(dataset)
        print("Total Target Samples: "+str(len(target_samples)))
        print("Err: "+str(err))
        wr = np.array(wr)
        ws = np.array(ws)
        print("Mean wr: "+str(np.mean(wr)))
        print("Mean ws: "+str(np.mean(ws)))
        
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

        sa,est = fqi.partial_fit(sast, r,**fit_params)
        Q1 = est.predict(sa)

        fit_params = {'sample_weight': ws}
        
        iterations = 60
        best_j = -float("Inf")
        best_policy = fqi
        for i in range(iterations - 1):
            fqi.partial_fit(None, Q1,**fit_params)
            values = evaluation.evaluate_policy(mdp, fqi, n_episodes = 1, initial_states = np.array([[100.0,1]]))
            if values[0] > best_j:
                best_policy = fqi
                best_j = values[0]
        evals.append(best_j)
        print(str(best_j))
        
    print(list(evals))
    #save_object(best_policy,'source_policy_dam1.pkl')
    json.dump(evals,perf_file)