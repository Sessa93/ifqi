import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.wfqi import WFQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.regressor import Regressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import Ensemble
import scipy.stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class Task:
    def __init__(self,name,data):
        self.name = name
        self.data = data

def labelSample(task, tasks, sample):
    eps = 0.5

    reward_idx = 3
    sast = np.append(task.data[:, :reward_idx],
                      task.data[:, reward_idx + reward_dim:reward_idx + reward_dim+3],
                      axis=1)
    R = task.data[:, reward_idx]
    SAS = sast[:,:5]
    rws = []
    c = 0
    total = 0

    for i in range(len(SAS)):
        if SAS[i,0] == sample[0] and SAS[i,1] == sample[1] and SAS[i,2] == sample[2]:
            rws.append(R[i])
            if SAS[i,3] == sample[4] and SAS[i,4] == sample[5]:
                c += 1
            total += 1

    std = np.std(rws)
    m = sample[3]

    if std == 0:
        if m == np.mean(rws):
            pr = 1
        else:
            pr = 0
    else:
        pr = scipy.stats.norm(loc=np.mean(rws), scale=std).cdf(m+eps) - scipy.stats.norm(loc=np.mean(rws), scale=std).cdf(m-eps)


    if total > 0:
        pp = c/total
    else:
        pp = 0
        print("Something Wrong!")

    QP = []
    QR = []
    for tsk in tasks:
        if not(tsk.name == task.name):
            reward_idx = 3
            sast = np.append(tsk.data[:, :reward_idx],
                              tsk.data[:, reward_idx + reward_dim:reward_idx + reward_dim+3],
                              axis=1)
            R = tsk.data[:, reward_idx]
            SAS = sast[:,:5]
            #print(sample)
            rws = []
            c = 0
            total = 0
            for i in range(len(SAS)):
                if SAS[i,0] == sample[0] and SAS[i,1] == sample[1] and SAS[i,2] == sample[2]:
                    rws.append(R[i])
                    if SAS[i,3] == sample[4] and SAS[i,4] == sample[5]:
                        c += 1
                    total += 1

            if len(rws) > 0:
                std = np.std(rws)
                if std == 0:
                    if m == np.mean(rws):
                        q = 1
                    else:
                        q = 0
                else:
                    q = scipy.stats.norm(loc=np.mean(rws), scale=std).cdf(m+eps) - scipy.stats.norm(loc=np.mean(rws), scale=std).cdf(m-eps)
            else:
                q = 0
            QR.append(q)

            if total > 0:
                QP.append(c/total)
            else:
                QP.append(0)

    qp = max(QP)
    qr = max(QR)
    #print(qp)

    wp = qp/pp
    wr = qr/pr


    #print([wp,wr])
    return [sample[0],sample[1],sample[2],sample[3],sample[4],sample[5],sample[6],sample[7],wp,wr]

"""
Simple script to quickly run fqi. It solves the Acrobot environment according
to the experiment presented in:

Ernst, Damien, Pierre Geurts, and Louis Wehenkel.
"Tree-based batch mode reinforcement learning."
Journal of Machine Learning Research 6.Apr (2005): 503-556.
"""

np.random.seed(32)

target_mdp = envs.SimpleGrid(np.array([0,2]),np.array([4,1]),10,5,eps=0)
source1_mdp = envs.SimpleGrid(np.array([0,2]),np.array([4,1]),10,5,eps=0.05) # Very similar
source2_mdp = envs.SimpleGrid(np.array([0,2]),np.array([4,1]),10,5,eps=0.1) # Not so similar
source3_mdp = envs.SimpleGrid(np.array([0,2]),np.array([4,1]),10,5,eps=0.1) # Definitely not similar
source4_mdp = envs.SimpleGrid(np.array([0,3]),np.array([4,1]),10,5,eps=0.1)
source5_mdp = envs.SimpleGrid(np.array([0,2]),np.array([4,1]),10,5,eps=0.05)


state_dim, action_dim, reward_dim = envs.get_space_info(target_mdp)
assert reward_dim == 1
regressor_params = {'n_estimators': 30,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}

discrete_actions = target_mdp.action_space.values
#print(discrete_actions)

# ExtraTrees
regressor = Regressor(ExtraTreesRegressor, **regressor_params)

# Action regressor of Ensemble of ExtraTreesEnsemble
#regressor = Ensemble(ExtraTreesRegressor, **regressor_params)
regressor = ActionRegressor(regressor, discrete_actions=discrete_actions, tol=.1)

data_target = Task('target',evaluation.collect_episodes(target_mdp, n_episodes=3))
data_s1 = Task('Source_1',evaluation.collect_episodes(source1_mdp, n_episodes=5))
data_s2 = Task('Source_2',evaluation.collect_episodes(source2_mdp, n_episodes=5))
data_s3 = Task('Source_3',evaluation.collect_episodes(source3_mdp, n_episodes=5))
data_s4 = Task('Source_4',evaluation.collect_episodes(source4_mdp, n_episodes=5))
data_s5 = Task('Source_5',evaluation.collect_episodes(source5_mdp, n_episodes=5))
tasks = [data_target,data_s1,data_s2,data_s3,data_s4,data_s5]

print("Target size: "+str(len(data_target.data)))
#print("Source size: "+str(len(data_s1.data)+len(data_s2.data)+len(data_s3.data)))

"""
tsk = data_s3
weights = []
for sample in tsk.data:
    weights.append(labelSample(tsk,tasks,sample)[9])

n, bins, patches = plt.hist(weights, 40, facecolor='red')
plt.title('Reward weights of ' + tsk.name + ', average')
plt.savefig('avg3r.eps', format='eps', dpi=1200)
plt.show()
"""

dataset = []

for t in tasks:
    if not(t.name == 'target'):
        for sample in t.data:
            dataset.append(labelSample(t,tasks,sample))

for sample in data_target.data:
    dataset.append([sample[0],sample[1],sample[2],sample[3],sample[4],sample[5],sample[6],sample[7],1,1])
dataset = np.array(dataset)

print("Total Samples: "+str(len(dataset)))

#check_dataset(dataset, state_dim, action_dim, reward_dim) # this is just a
# check, it can be removed in experiments
#print('Dataset has %d samples' % dataset.shape[0])


reward_idx = state_dim + action_dim
sast = np.append(dataset[:, :reward_idx],
                  dataset[:, reward_idx + reward_dim:reward_idx + reward_dim+3],
                  axis=1)
r = dataset[:, reward_idx]
wp = dataset[:, reward_idx+5]
wr = dataset[:, reward_idx+6]
#sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

fqi_iterations = target_mdp.horizon  # this is usually less than the horizon
wfqi = WFQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=target_mdp.gamma,
          horizon=fqi_iterations,
          verbose=True)

fit_params = {}
#fit_params = {
#     "n_epochs": 300,
#     "batch_size": 50,
#     "validation_split": 0.1,
#     "verbosity": False,
#     "criterion": "mse"
#}

wfqi.partial_fit(sast,r,wp,wr, **fit_params)
values = evaluation.evaluate_policy(target_mdp, wfqi, n_episodes=10)
print(values)


iterations = 30
transfer = []
#plt.title('Performance with transfer - average case')
plt.xlabel('WFQI iterations')
plt.ylabel('Discounted expected reward')
for i in range(iterations - 1):
    wfqi.partial_fit(None, None, None, None, **fit_params)

    values = evaluation.evaluate_policy(target_mdp, wfqi, n_episodes=15,horizon=100)
    print(values)
    transfer.append(values[0])
    """
    if i == 1:
        fig1 = plt.figure(1)
        ax = fig1.add_subplot(1, 1, 1)
        h = ax.plot(range(i + 1), iteration_values, 'ro-')
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.ion()  # turns on interactive mode
        plt.show()
    elif i > 1:
        h[0].set_data(range(i + 1), iteration_values)
        ax.figure.canvas.draw()
        plt.ylim(min(iteration_values), max(iteration_values))
        plt.xlim(0, i + 1)
        plt.show()
    """

print("NO TRANSFER")
wfqi = WFQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=target_mdp.gamma,
          horizon=fqi_iterations,
          verbose=True)
no_transfer = []
reward_idx = state_dim + action_dim
dataset = data_target.data
sast = np.append(dataset[:, :reward_idx],
                  dataset[:, reward_idx + reward_dim:reward_idx + reward_dim+3],
                  axis=1)
r = dataset[:, reward_idx]
wp = 1
wr = 1
wfqi.partial_fit(sast,r,wp,wr, **fit_params)
values = evaluation.evaluate_policy(target_mdp, wfqi, n_episodes=10)
print(values)
for i in range(iterations - 1):
    wfqi.partial_fit(None, None, None, None, **fit_params)

    values = evaluation.evaluate_policy(target_mdp, wfqi, n_episodes=15,horizon=100)
    print(values)
    no_transfer.append(values[0])
#

line1, = plt.plot(list(range(iterations-1)),transfer,'ro-',label='WFQI')
line2, = plt.plot(list(range(iterations-1)),no_transfer,'b+-',label='FQI')

plt.legend(handles=[line1,line2], loc=4)
plt.savefig('perfMAX.eps', format='eps', dpi=1200)
plt.show()
s = input()
