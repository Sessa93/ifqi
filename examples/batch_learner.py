import sys
sys.path.append("/home/transfer/ifqi")
#sys.path.append("/Users/andrea/Desktop/ifqi")

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

delta_sa = 0.1
delta_r = 0.5
delta_p = 0.1
mu = 0.8

def distance_sa(s1,s2,a1,a2):
    return math.sqrt(math.pow((s1[0]-s2[0]),2) + math.pow((s1[1]-s2[1]),2) + math.pow((a1-a2),2))

def distance(s1,s2):
    return math.sqrt( math.pow((s1[0]-s2[0]),2) + math.pow((s1[1]-s2[1]),2))

def phi(x,delta):
    return np.exp(-math.pow(x,2)/(delta))

# Compliance calcolate
# Source1 -> 0.5167
# Source2 -> 0.1036
# Source3 -> 0.3797
def compliance(target_samples, source_samples):
    prior = 1/3
    n_target = len(target_samples)
    n_source = len(source_samples)
    lambdas_i = []
    count = 0

    for (sxt,syt,at,rt,nxt,nyt,f1,f2) in target_samples:
        lambdas_ij_p = []
        lambdas_ij_r = []
        for (sxs,sys,ass,rs,nxs,nys,f3,f4) in source_samples:
            tmp_sum = 0
            for (sxl,syl,al,rl,nxl,nyl,f5,f6) in source_samples:
                tmp_sum += phi(distance_sa([sxt,syt],[sxl,syl],at,al),delta_sa)

            #print('Tmp_sum: ',str(tmp_sum))

            w_ij = phi(distance_sa([sxt,syt],[sxs,sys],at,ass),delta_sa)/tmp_sum
            #print('wij: ',str(w_ij))
            lambdas_ij_p.append(w_ij*phi(distance([nxt,nyt],[sxt+(nxs-sxs), syt+(nys-sys)]),delta_p))
            lambdas_ij_r.append(w_ij*phi(rt-rs,delta_r))

        lambdas_ij_p = np.array(lambdas_ij_p)
        lambdas_ij_r = np.array(lambdas_ij_r)

        lambda_ij_p = np.mean(lambdas_ij_p)
        lambda_ij_r = np.mean(lambdas_ij_r)

        lambdas_i.append(lambda_ij_p*lambda_ij_r)
        #print(lambda_ij_p*lambda_ij_r)
        if count % 100 == 0:
            print(count)
        count += 1

    result = 0

    for t in range(n_target):
        result += lambdas_i[t]*prior
    return result/n_target

def calculate_h(omegas):
    total = 0
    i = 1
    omegas.sort()
    while total < mu and i < len(omegas):
        total += omegas[i]
        i += 1
    return i

# Dati i samples di target e source, arrichisce i source con la relevance e salva il dataset
def relevance(target_samples, source_samples):
    n_target = len(target_samples)
    n_source = len(source_samples)

    samples = []
    d = []
    count = 0
    sums_p = []
    sums_r = []
    lambda_j = []

    tmp_sums = []
    for (sxs,sys,ass,rs,nxs,nys,f1,f2) in source_samples:
        tmp_sum = 0.0001
        for (sxl,syl,al,rl,nxl,nyl,f5,f6) in target_samples:
                tmp_sum += phi(distance_sa([sxs,sys],[sxl,syl],ass,al),delta_sa)
        tmp_sums.append(tmp_sum)

    j = 0
    for (sxs,sys,ass,rs,nxs,nys,f1,f2) in source_samples:
        lambdas_ij_p = []
        lambdas_ij_r = []

        t_omega = []
        omegas = []
        for (sxt,syt,at,rt,nxt,nyt,f3,f4) in target_samples:

            w_ij = phi(distance_sa([sxs,sys],[sxt,syt],at,ass),delta_sa) /tmp_sums[j]
            lambdas_ij_p.append(w_ij*phi(distance([nxs,nys],[sxs+(nxt-sxt), sys+(nyt-syt)]),delta_p))
            lambdas_ij_r.append(w_ij*phi(abs(rt-rs),delta_r))
            t_omega.append([w_ij,sxt,syt,at,rt,nxt,nyt,f3,f4])
            omegas.append(w_ij)

        h = calculate_h(omegas)
        #print(h)
        t_omega.sort(key=(lambda x:x[0]))

        total_d = 0
        for i in range(h):
            (w_ij,sxt,syt,at,rt,nxt,nyt,f3,f4) = t_omega[i]
            total_d += distance_sa([sxs,sys],[sxt,syt], ass, at)
        d.append(total_d / h)

        lambda_sum_p = sum(lambdas_ij_p)
        lambda_sum_r = sum(lambdas_ij_r)
        sums_p.append(lambda_sum_p)
        sums_r.append(lambda_sum_r)
        j += 1

    j = 0
    Z_p = sum(sums_p)
    Z_r = sum(sums_r)
    for (sxs,sys,ass,rs,nxs,nys,f1,f2) in source_samples:
        lambda_j.append((sums_p[j]/Z_p)*(sums_r[j]/Z_r))
        j += 1

    s = 0
    for (sxs,sys,ass,rs,nxs,nys,f1,f2) in source_samples:
        norm_l = lambda_j[s]/sum(lambda_j)
        rho = np.exp(-math.pow((norm_l-1)/d[s],2))
        samples.append([rho,sxs,sys,ass,rs,nxs,nys,f1,f2])
        s += 1
    return samples



n_source = 4000
for n_target in [5500]:
    evals = []
    for e in range(1):
        samples_source1 = []
        samples_source2 = []
        samples_source3 = []
        samples_target = []

        err = 0
        # SAMPLES COLLECTION -----------------------
        samples_target = evaluation.collect_episodes(target_mdp,n_episodes=int(5000/50))
        samples_target = samples_target[:5000]

        best_policy_source = load_object('source_policy_1.pkl')
        samples_source1 = evaluation.collect_episodes(source_mdp_1, best_policy_source, n_episodes=int(n_source/50))
        samples_source1 = samples_source1[:1000]

        best_policy_source = load_object('source_policy_2.pkl')
        samples_source2 = evaluation.collect_episodes(source_mdp_2, best_policy_source, n_episodes=int(n_source/50))
        samples_source2 = samples_source2[:1000]

        best_policy_source = load_object('source_policy_3.pkl')
        samples_source3 = evaluation.collect_episodes(source_mdp_3, best_policy_source, n_episodes=int(n_source/50))
        samples_source3 = samples_source3[:1000]


        #samples_target = [[0,0,1,-1,1,1,0,0]]
        #samples_source = [[0,0,1,-1,1,1,0,0]]

        print('Calculating relevance 1')
        save_object(relevance(samples_target, samples_source3),'dataset_3.pkl')

        # END SAMPLES LABELING -------------------
