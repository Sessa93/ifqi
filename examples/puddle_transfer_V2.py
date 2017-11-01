import sys
sys.path.append("/home/transfer/ifqi") 'Va modificato con il path giusto!

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


perf_file = open('perf_500_M_V2_SEL.txt', 'w')

# Tasks definition
source_mdp_1 = envs.PuddleWorld2(goal_x=5,goal_y=10)
source_mdp_2 = envs.PuddleWorld2(goal_x=5,goal_y=10, puddle_means=[(2.0,2.0),(4.0,6.0),(1.0,8.0), (2.0, 4.0), (8.5,7.0),(8.5,5.0)], puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)])
source_mdp_3 = envs.PuddleWorld2(goal_x=7,goal_y=10, puddle_means=[(8.0,2.0), (1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)])
source_mdps = [source_mdp_1,source_mdp_2,source_mdp_3]
target_mdp = envs.PuddleWorld2(goal_x=5,goal_y=10, puddle_means=[(1.0,4.0),(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], puddle_var=[(.7, 1.e-5, 1.e-5, .7), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)])


state_dim, action_dim, reward_dim = envs.get_space_info(source_mdp_1) # Same for the target task
assert reward_dim == 1
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split':2,
                    'min_samples_leaf': 1,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = source_mdp_1.action_space.values

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
# regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions, tol=.1)

'La funzione collect espisode raccoglie traiettorie, bisogna specificare il numero corretto di traiettorie!
n_source = 4000 
for n_target in [500, 1000, 2000, 3000, 4000, 5000]:
    evals = []
    for e in range(10):
        err = 0
        k = 1
        # SAMPLES LABELING -----------------------
        target_samples = evaluation.collect_episodes(target_mdp,n_episodes=int(n_target/50))
        sast, r = split_data_for_fqi(target_samples, state_dim, action_dim, reward_dim)
        dataset = []
        if n_source > 0:
            X = sast[:,0:3]
            y = r
            gp_reward_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_reward_target.fit(X,y)

            print("Target task reward GP fitted!")

            y = sast[:,3]
            gp_state_x_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_state_x_target.fit(X,y)

            y = sast[:,4]
            gp_state_y_target = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
            gp_state_y_target.fit(X,y)
            print("Target task transition GP fitted!")

            for mdp in source_mdps:
                print("Started samples labeling...")

                best_policy_source = load_object('source_policy_'+str(k)+'.pkl')
                source_samples = evaluation.collect_episodes(mdp, best_policy_source, n_episodes=int((n_source/50)))
                sast, r = split_data_for_fqi(source_samples, state_dim, action_dim, reward_dim)

                Xs = sast[:,0:3]
                y = r
                gp_reward_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_reward_source.fit(Xs,y)
                mu_s, std_s = gp_reward_source.predict(Xs,return_std=True)
                print("Source task "+str(k)+" reward GP fitted!")

                y = sast[:,3]
                gp_state_x_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_state_x_source.fit(Xs,y)
                mu_s_x, std_s_x = gp_state_x_source.predict(Xs,return_std=True)

                y = sast[:,4]
                gp_state_y_source = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.1)
                gp_state_y_source.fit(Xs,y)
                mu_s_y, std_s_y = gp_state_y_source.predict(Xs,return_std=True)
                print("Source task "+str(k)+" transition GP fitted!")

                mu_t, std_t = gp_reward_target.predict(Xs,return_std=True)
                mu_t_x, std_t_x = gp_state_x_target.predict(Xs,return_std=True)
                mu_t_y, std_t_y = gp_state_y_target.predict(Xs,return_std=True)

                N = 50 # Number of samples to sample

                raw_dataset = []
                for i in range(len(source_samples)):
                    (x,y,a,r,xs,ys,f1,f2) = source_samples[i]

                    v1r = math.pow(std_t[i],2)+0.01
                    v2r = 0.01-math.pow(std_s[i],2)

                    v1x = math.pow(std_t_x[i],2)+0.04
                    v2x = 0.04-math.pow(std_s_x[i],2)

                    v1y = math.pow(std_t_y[i],2)+0.04
                    v2y = 0.04-math.pow(std_s_y[i],2)
                    
                    if v2r > 0 and v2x > 0 and v2y > 0:
                        d_r = stats.norm.pdf(r, mu_s[i], math.sqrt(v2r))
                        n_r = stats.norm.pdf(r, mu_t[i], math.sqrt(v1r))
                        wr = (n_r/d_r)*(0.01/v2r)
                       
                        drx = stats.norm.pdf(xs, mu_s_x[i], math.sqrt(v2x))
                        nrx = stats.norm.pdf(xs, mu_t_x[i], math.sqrt(v1x))

                        dry = stats.norm.pdf(ys, mu_s_y[i], math.sqrt(v2y))
                        nry = stats.norm.pdf(ys, mu_t_y[i], math.sqrt(v1y))

                        ws = (nrx/drx)*(nry/dry)*(0.04/v2y)*(0.04/v2x)
                        
                        raw_dataset.append([x,y,a,r,xs,ys,f1,f2,ws,wr])
                    else:
                        err += 1
                # Weight filtering
                for (x,y,a,r,xs,ys,f1,f2,ws,wr) in raw_dataset:
                    if ws <= 10 and wr <= 10:
                        dataset.append([x,y,a,r,xs,ys,f1,f2,ws,wr])
                del raw_dataset
                k += 1

        print("Total Source Samples: "+str(len(dataset)))

        # Add target samples with unitary weights
        if n_source == 0:
            dataset = []
        for (x,y,a,r,xs,ys,f1,f2) in target_samples:
            dataset.append([x,y,a,r,xs,ys,f1,f2,1.0,1.0])
        print("Target Samples: "+str(len(target_samples)))
        dataset = np.array(dataset)

        # END SAMPLES LABELING -------------------

        #print(state_dim)
        #print(action_dim)

        reward_idx = state_dim + action_dim
        sast = np.append(dataset[:, :reward_idx],
                          dataset[:, reward_idx + reward_dim:reward_idx + reward_dim+3],
                          axis=1)
        r = dataset[:, reward_idx]
        ws = dataset[:, reward_idx+5]
        wr = dataset[:,reward_idx+6]

        print("Sw: "+str(np.mean(ws)))
        print("Rw: "+str(np.mean(wr)))
        print("Err: "+str(err))

        fqi_iterations = target_mdp.horizon  # this is usually less than the horizon
        fqi = FQI(estimator=regressor,
                  state_dim=state_dim,
                  action_dim=action_dim,
                  discrete_actions=discrete_actions,
                  gamma=target_mdp.gamma,
                  horizon=fqi_iterations,
                  verbose=False)

        fit_params = {'sample_weight': wr}

        sa, est = fqi.partial_fit(sast, r, **fit_params)

        fit_params = {'sample_weight': ws}

        Q1 = est.predict(sa)

        n_initial = 1
        initial_states = []
        for i in range(n_initial):
            x = np.random.random()
            y = np.random.random()
            initial_states.append([x,y])
        initial_states = np.array(initial_states)

        iterations = 30
        iteration_values = []
        best_j = -100
        best_policy = fqi
        for i in range(iterations - 1):
            fqi.partial_fit(None, Q1, **fit_params)
            values = evaluation.evaluate_policy(target_mdp, fqi,horizon=50,initial_states=initial_states, n_episodes=1)
            #print(values)
            iteration_values.append(values[0])
            if values[0] > best_j:
                best_policy = fqi
                best_j = values[0]
                steps = values[2]
        evals.append(best_j)
        print(str(best_j)+' '+str(steps))
    print(list(evals))
    json.dump(evals,perf_file)
