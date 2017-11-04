import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.regressor import Regressor

mdp = envs.Dam(demand = 50.0, flooding = 50.0, inflow_mean = 40.0, inflow_std = 10, alpha = 0.4, beta = 0.6)
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)
assert reward_dim == 1

regressor_params = {'n_estimators': 20,
                    'criterion': 'mse',
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = mdp.action_space.values

regressor = Regressor(ExtraTreesRegressor, **regressor_params)

dataset = evaluation.collect_episodes(mdp, n_episodes=100)
check_dataset(dataset, state_dim, action_dim, reward_dim)

print('Dataset has %d samples' % dataset.shape[0])

sast, r = split_data_for_fqi(dataset, state_dim, action_dim, reward_dim)

fqi_iterations = mdp.horizon
fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=fqi_iterations,
          verbose=True)

fit_params = {}

fqi.partial_fit(sast, r, **fit_params)

iterations = 20
iteration_values = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fit_params)
    values = evaluation.evaluate_policy(mdp, fqi, n_episodes = 20)
    print(values)
    iteration_values.append(values[0])

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
        

input("Press Enter to exit...")
