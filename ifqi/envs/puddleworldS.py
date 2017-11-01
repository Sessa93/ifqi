# Inspired by https://github.com/amarack/python-rl/blob/master/pyrl/environments/puddleworld.py

import gym
import numpy as np
from builtins import range
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
from scipy.integrate import odeint
import math
import ifqi.utils.spaces as fqispaces



register(
    id='PuddleWorld-v1',
    entry_point='ifqi.envs.PuddleWorld:PuddleWorldS'
)

class PuddleWorldS(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, size_x=10, size_y=10, goal_x=10, goal_y=10, puddle_penalty=30.0,
             puddle_means=[(1.0, 10.0), (1.0, 8.0), (6.0,6.0),(6.0,4.0)], puddle_var=[(.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8), (.8, 1.e-5, 1.e-5, .8),(.8, 1.e-5, 1.e-5, .8)],
             noise=0.2, reward_noise=0.1, fudge=1.41):

        self.size = np.array([size_x,size_y])
        self.goal = np.array([goal_x,goal_y])
        self.noise = noise
        self.reward_noise = reward_noise
        self.fudge = fudge

        self.puddle_penalty = puddle_penalty
        self.puddle_means = list(map(np.array, puddle_means))

        self.puddle_var = list(map(lambda cov: np.linalg.inv(np.array(cov).reshape((2,2))), puddle_var))
        self.puddles = list(zip(self.puddle_means, self.puddle_var))

        self.horizon = 100
        self.viewer = None
        self.gamma = 0.99

        high = np.array([size_x, size_y])
        self.observation_space = spaces.Box(low=np.array([0,0]), high=high)

        self.action_space = fqispaces.DiscreteValued([0.,1.,2.,3.], decimals=0)

        self.pos = np.array([0., 0.])
        self.initial_states = np.array([self.pos])

        self.seed()
        self.reset()

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            self.pos = np.array([0., 0.])
        else:
            self.pos = np.array([0., 0.])
        return np.array(self.pos)

    def isAtGoal(self):
        return np.linalg.norm(self.pos - self.goal) < self.fudge

    def posIsAtGoal(self, pos):
        return np.linalg.norm(pos - self.goal) < self.fudge

    def mvnpdf(self, x, mu, sigma_inv):
        size = len(x)
        if size == len(mu) and sigma_inv.shape == (size, size):
            det = 1./np.linalg.det(sigma_inv)
            norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,0.5) )
            x_mu = x - mu
            result = math.pow(math.e, -0.5 * np.dot(x_mu, np.dot(sigma_inv, x_mu)))
            return norm_const * result
        else:
            raise NameError("The dimensions of the input don't match")

    def step(self, a):

        slow_factor = 0
        #print(list(zip(self.puddle_means, self.puddle_var)))
        for mu, inv_cov in self.puddles:
            slow_factor += self.mvnpdf(self.pos, mu, inv_cov)
        #print(base_reward)

        alpha = 1/(1+(self.puddle_penalty*slow_factor))

        if int(a) == 0:
            self.pos[0] += alpha
        elif int(a) == 1:
            self.pos[0] -= alpha
        elif int(a) == 2:
            self.pos[1] += alpha
        elif int(a) == 3:
            self.pos[1] -= alpha

        if self.noise > 0:
            self.pos += np.random.normal(scale=self.noise, size=(2,))
        self.pos = self.pos.clip([0,0],self.size)

        base_reward = 0.0 if self.isAtGoal() else -1.0
        if self.reward_noise > 0:
            base_reward += np.random.normal(scale=self.reward_noise)

        if self.isAtGoal():
            self._absorbing = True
        else:
            self._absorbing = False

        #print(base_reward)
        return self.pos, base_reward, self._absorbing, {}

    def get_state(self):
        return self.pos

    def _render(self, mode=None, close=None):
        pass

    def getNextState(self,pos,a):
        slow_factor = 0
        for mu, inv_cov in self.puddles:
            slow_factor += self.mvnpdf(pos, mu, inv_cov)
        alpha = 1/(1+(self.puddle_penalty*slow_factor))
        if int(a) == 0:
            pos[0] += alpha
        elif int(a) == 1:
            pos[0] -= alpha
        elif int(a) == 2:
            pos[1] += alpha
        elif int(a) == 3:
            pos[1] -= alpha
        return pos
