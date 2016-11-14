import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.fqi.FQI import FQI
from ifqi.models.actionregressor import ActionRegressor
from ifqi.models.mlp import MLP
from ifqi.models.ensemble import ExtraTreesEnsemble

mdp = envs.CarOnHill()
state_dim, action_dim = envs.get_space_info(mdp)
regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 4,
                    'min_samples_leaf': 2}
discrete_actions = mdp.action_space.values
regressor = ExtraTreesEnsemble
# regressor = MLP(3, 1, [15], 'relu', 'rmsprop')

regressor = ActionRegressor(regressor,
                            discrete_actions=discrete_actions, decimals=5,
                            **regressor_params)

dataset = evaluation.collect_episodes(mdp, n_episodes=10)

reward_idx = state_dim + action_dim
sast = np.append(dataset[:, :reward_idx], dataset[:, reward_idx + 1:], axis=1)
r = dataset[:, reward_idx]

fqi = FQI(estimator=regressor,
          state_dim=state_dim,
          action_dim=action_dim,
          discrete_actions=discrete_actions,
          gamma=mdp.gamma,
          horizon=mdp.horizon,
          scaled=False,
          features=None,
          verbose=True)

fitParams = {}
# fitParams = {
#     "nEpochs": 300,
#     "batchSize": 50,
#     "validationSplit": 0.1,
#     "verbosity": False,
#     "criterion": "mse"
# }

initial_states = np.zeros((289, 2))
count = 0
for i in range(-8, 9):
    for j in range(-8, 9):
        initial_states[count, :] = np.array([0.125 * i, 0.375 * j])
        count += 1

fqi.partial_fit(sast, r, **fitParams)

iterations = 20
iteration_values = []
for i in range(iterations - 1):
    fqi.partial_fit(None, None, **fitParams)
    values = evaluation.evaluate_policy(mdp, fqi,
                                        initial_states=initial_states,
                                        n_episodes=289)
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
