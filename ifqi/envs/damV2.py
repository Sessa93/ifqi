import gym
import numpy as np
from gym import spaces
from gym.spaces import prng
import ifqi.utils.spaces as fqispaces

"""
Dam Control (discretized action space)

References
----------
  - Simone Parisi, Matteo Pirotta, Nicola Smacchia,
    Luca Bascetta, Marcello Restelli,
    Policy gradient approaches for multi-objective sequential decision making
    2014 International Joint Conference on Neural Networks (IJCNN)

"""

from gym.envs.registration import register

register(
    id='Dam-v2',
    entry_point='ifqi.envs.damV2:Dam',
    timestep_limit=300,
)


class Dam(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, demand = 50.0, flooding = 50.0, inflow_mean = 40.0, inflow_std = 10, alpha = 0.5, beta = 0.5):
        
        self.horizon = 100
        self.gamma = 1.0

        self.DEMAND = demand  # Water demand -> At least DEMAND/day must be supplied or a cost is incurred
        self.FLOODING = flooding  # Flooding threshold -> No more than FLOODING can be stored or a cost is incurred
        # NOTE: we do not allow to change the capacity since it would change the action space and the initial state distribution
        self.CAPACITY = 100.0  # Release threshold (i.e., max capacity) -> At least max{S - CAPACITY, 0} must be released
        self.INFLOW_MEAN = inflow_mean  # Random inflow (e.g. rain) mean
        self.INFLOW_STD = inflow_std # Random inflow std
        
        assert alpha + beta == 1.0 # Check correctness
        self.ALPHA = alpha # Weight for the flooding cost
        self.BETA = beta # Weight for the demand cost
        
        # Gym attributes
        self.viewer = None
        
        self.n_actions = 11
        actions = [float(i) / (self.n_actions - 1) * self.CAPACITY for i in range(self.n_actions)]
        
        self.action_space = fqispaces.DiscreteValued(actions, decimals=0)   
        
        self.observation_space = spaces.Box(low=0.0,
                                            high=np.inf,
                                            shape=(1,))

        # Initialization
        self.seed()
        self.reset()

    def step(self, action, render=False):
        
        # Get current state
        state = self.get_state()
        
        # Bound the action
        actionLB = max(state - self.CAPACITY, np.array([0.0]))
        actionUB = state

        # Penalty proportional to the violation
        bounded_action = min(max(action, actionLB), actionUB)
        penalty = -abs(bounded_action - action)

        # Transition dynamics
        action = bounded_action
        dam_inflow = self.INFLOW_MEAN + np.random.randn() * self.INFLOW_STD
        nextstate = max(state + dam_inflow - action, np.array([0.0]))

        # Cost due to the excess level wrt the flooding threshold
        reward_flooding = -max(nextstate - self.FLOODING, np.array([0.0])) + penalty

        # Deficit in the water supply wrt the water demand
        reward_demand = -max(self.DEMAND - action, np.array([0.0])) + penalty
        
        # The final reward is a weighted average of the two costs
        reward = self.ALPHA * reward_flooding + self.BETA * reward_demand

        self.state = nextstate

        return self.get_state(), np.asscalar(reward), False, {}

    def reset(self, state=None):
        
        if state is None:
            self.state = [prng.np_random.uniform(0.0, self.CAPACITY * 1.2)]
        else:
            assert np.isscalar(state) and state > 0.
            self.state = [np.asscalar(state)]

        return self.get_state()

    def get_state(self):
        return np.array(self.state)