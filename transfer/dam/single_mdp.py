import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from ifqi import envs
from ifqi.evaluation import evaluation
from ifqi.evaluation.utils import check_dataset, split_data_for_fqi
from ifqi.algorithms.fqi import FQI
from ifqi.models.regressor import Regressor
import json

perf_file = open('perf_dam.txt', 'w')

mdp = envs.Dam()
state_dim, action_dim, reward_dim = envs.get_space_info(mdp)

regressor_params = {'n_estimators': 50,
                    'criterion': 'mse',
                    'min_samples_split': 20,
                    'min_samples_leaf': 2,
                    'input_scaled': False,
                    'output_scaled': False}
discrete_actions = mdp.action_space.values

regressor = Regressor(ExtraTreesRegressor, **regressor_params)

for n_samples in [1,5,10,20,30,40,50,100,300]:
    
    print("Starting N = " + str(n_samples))
    
    evals = [n_samples]
    
    for e in range(20):

        dataset = evaluation.collect_episodes(mdp, n_episodes=n_samples)
        check_dataset(dataset, state_dim, action_dim, reward_dim)

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

        iterations = 60
        best_j = -float("Inf")
        best_policy = fqi
        for i in range(iterations - 1):
            fqi.partial_fit(None, None, **fit_params)
            values = evaluation.evaluate_policy(mdp, fqi, n_episodes = 1, initial_states = np.array([[100.0,1]]))
            if values[0] > best_j:
                best_policy = fqi
                best_j = values[0]
        evals.append(best_j)
        print(str(best_j))
        
    print(list(evals))
    json.dump(evals,perf_file)
