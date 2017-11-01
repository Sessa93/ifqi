import sys
sys.path.append("/home/transfer/ifqi")
#sys.path.append("/Users/andrea/Desktop/Dropbox/Appunti/Thesis/ifqi/ifqi")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.algorithms.wfqi import WFQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import Ensemble
import scipy.stats as stats
from matplotlib.mlab import bivariate_normal
import math
import matplotlib.mlab as mlab
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


perf_file = open('perf_500_M_acro_MEAN.txt', 'w')
# Tasks definition
source_mdp_1 = envs.Acrobot(0.8,0.6,0.9,1.,0.1,0.1)
source_mdp_2 = envs.Acrobot(0.95,0.95,0.95,1.,0.1,0.1)
source_mdp_3 = envs.Acrobot(0.85,0.85,0.9,0.9,0.1,0.1)
target_mdp = envs.Acrobot(1.,1.,1.,1.,.1,.1)

source_mdps = [source_mdp_1, source_mdp_2,source_mdp_3]

state_dim, action_dim, reward_dim = envs.get_space_info(source_mdp_1) # Same for the target task
assert reward_dim == 1
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = source_mdp_1.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
# regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions, tol=.1)

n_source = 500
for n_target in [200,500,1000,2000,3000,4000,5000]:
    evals = []
    for e in range(30):
        W = []
        wr = []
        err = 0
        print('Test no. '+str(e+1))
        k = 1
        # SAMPLES LABELING -----------------------
        target_samples = evaluation.collect_episodes(target_mdp,n_episodes=int(n_target/100))
        sast, r = split_data_for_fqi(target_samples, state_dim, action_dim, reward_dim)

        dataset = []
        if n_source > 0:
            X = sast[:,0:5]

            y = sast[:,5]
            gp_theta1_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_theta1_target.fit(X,y)

            y = sast[:,6]
            gp_theta2_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_theta2_target.fit(X,y)

            y = sast[:,7]
            gp_thetadot1_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_thetadot1_target.fit(X,y)

            y = sast[:,8]
            gp_thetadot2_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_thetadot2_target.fit(X,y)
            print("Target task transition GP fitted!")

            for mdp in source_mdps:
                print("Started samples labeling...")

                best_policy_source = load_object('source_policy_acro'+str(k)+'.pkl')
                source_samples = evaluation.collect_episodes(mdp, best_policy_source, n_episodes=int((n_source/100)))
                sast, r = split_data_for_fqi(source_samples, state_dim, action_dim, reward_dim)

                X = sast[:,0:5]

                y = sast[:,5]
                gp_theta1_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_theta1_source.fit(X,y)

                y = sast[:,6]
                gp_theta2_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_theta2_source.fit(X,y)

                y = sast[:,7]
                gp_thetadot1_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_thetadot1_source.fit(X,y)

                y = sast[:,8]
                gp_thetadot2_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_thetadot2_source.fit(X,y)
                
                print("Source task "+str(k)+" transition GP fitted!")

                mu_t_theta1, std_t_theta1 = gp_theta1_target.predict(X,return_std=True)
                mu_t_theta2, std_t_theta2 = gp_theta2_target.predict(X,return_std=True)
                mu_t_thetadot1, std_t_thetadot1 = gp_thetadot1_target.predict(X,return_std=True)
                mu_t_thetadot2, std_t_thetadot2 = gp_thetadot2_target.predict(X,return_std=True)

                mu_s_theta1, std_s_theta1 = gp_theta1_source.predict(X,return_std=True)
                mu_s_theta2, std_s_theta2 = gp_theta2_source.predict(X,return_std=True)
                mu_s_thetadot1, std_s_thetadot1 = gp_thetadot1_source.predict(X,return_std=True)
                mu_s_thetadot2, std_s_thetadot2 = gp_thetadot2_source.predict(X,return_std=True)

                raw_dataset = []
                for i in range(len(source_samples)):
                    (t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2) = source_samples[i]
                    
                    sigma = 0.09
                    v11 = sigma + math.pow(std_t_theta1[i],2)
                    v12 = sigma - math.pow(std_s_theta1[i],2)
                    
                    v21 = sigma + math.pow(std_t_theta2[i],2)
                    v22 = sigma - math.pow(std_s_theta2[i],2)
                    
                    v31 = sigma + math.pow(std_t_thetadot1[i],2)
                    v32 = sigma - math.pow(std_s_thetadot1[i],2)
                                          
                    v41 = sigma + math.pow(std_t_thetadot2[i],2)
                    v42 = sigma - math.pow(std_s_thetadot2[i],2)
                    
                    if v12 > 0 and v22 > 0 and v32 > 0 and v42 > 0:
                        d1 = stats.norm.pdf(t1s, mu_s_theta1[i], math.sqrt(v12))
                        n1 = stats.norm.pdf(t1s, mu_t_theta1[i], math.sqrt(v11))

                        d2 = stats.norm.pdf(t2s, mu_s_theta2[i], math.sqrt(v22))
                        n2 = stats.norm.pdf(t2s, mu_t_theta2[i], math.sqrt(v21))

                        d3 = stats.norm.pdf(td1s, mu_s_thetadot1[i], math.sqrt(v32))
                        n3 = stats.norm.pdf(td1s, mu_t_thetadot1[i], math.sqrt(v31))

                        d4 = stats.norm.pdf(td2s, mu_s_thetadot2[i], math.sqrt(v42))
                        n4 = stats.norm.pdf(td2s, mu_t_thetadot2[i], math.sqrt(v41))

                        w = (n1/d1)*(n2/d2)*(n3/d3)*(n4/d4)*(0.01/v12)*(0.01/v22)*(0.01/v32)*(0.01/v42)
                    
                        raw_dataset.append([t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2,w])
                    else:
                        err += 1
                
                
                # Weight filtering
                for (t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2,ws) in raw_dataset:
                    if ws < 10:
                        dataset.append([t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2])
                        W.append(ws)
                        wr.append(1.)
                del raw_dataset
                k += 1

        print("Total Source Samples: "+str(len(dataset)))
        for (t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2) in target_samples:
            dataset.append([t1,t2,td1,td2,a,r,t1s,t2s,td1s,td2s,f1,f2])
            W.append(1.0)
            wr.append(1.)
        dataset = np.array(dataset)
        print("Total Target Samples: "+str(len(target_samples)))
        print("Err: "+str(err))

        print("W: "+str(np.mean(W)))

        sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

        W = np.array(W)
        wr = np.array(wr)
        fit_params = {'sample_weight': wr}
        
        fqi_iterations = target_mdp.horizon  # this is usually less than the horizon
        fqi = FQI(estimator=regressor,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  discrete_actions=discrete_actions,
                  gamma=target_mdp.gamma,
                  horizon=fqi_iterations,
                  verbose=False)

        sa,est = fqi.partial_fit(sast, r,**fit_params)
        Q1 = est.predict(sa)

        fit_params = {'sample_weight': W}
        
        iterations = 50
        iteration_values = []
        best_j = -100
        best_policy = fqi
        for i in range(iterations - 1):
            fqi.partial_fit(None, Q1,**fit_params)
            values = evaluation.evaluate_policy(target_mdp, fqi, horizon=100)
            #print(values)
            iteration_values.append(values[0])
            if values[0] > best_j:
                best_policy = fqi
                best_j = values[0]
                steps = values[2]
        evals.append(best_j)
        print(str(best_j)+' '+str(steps))
    print(list(evals))
    #save_object(best_policy,'source_policy_acro3.pkl')
    json.dump(evals,perf_file)
