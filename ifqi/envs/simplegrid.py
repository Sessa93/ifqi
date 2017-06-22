import gym
import numpy as np
from builtins import range
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
from scipy.integrate import odeint

import ifqi.utils.spaces as fqispaces


register(
    id='SimpleGrid-v0',
    entry_point='ifqi.envs.simpleGrid:SimpleGrid'
)


class SimpleGrid(gym.Env):
    """
    Simple grid-world environment with walls and 4 directions (N,S,E,W)

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self,start,goal,width=8,height=4,delta=0,eps=0):

        self.horizon = 60
        self.gamma = 0.95
        self.height = height
        self.width = width
        self.eps = eps
        self.walls = [(2,1),(2,2),(2,3),(3,2),(4,2),(5,2),(6,2),(7,1),(7,2),(7,3)]

        if start is None:
            self.start = np.array([0,2])
        else:
            self.start = start
        if goal is None:
            self.goal = np.array([9,2])
        else:
            self.goal = goal


        # gym attributes
        self.viewer = None
        high = np.array([self.width-1, self.height-1])
        self.observation_space = spaces.Box(low=np.array([0,0]), high=high)

        """
        Possible actions:
            0 -> N
            1 -> S
            2 -> E
            3 -> W
        """
        self.action_space = fqispaces.DiscreteValued([0.,1.,2.,3.], decimals=0)

        # evaluation initial states
        self.initial_states = np.array([start])
        #np.random.seed(42)

        # initialize state
        self.seed()
        self.reset()

    def step(self, u):
        sa = np.append(self._state, u)
        x = self._state[0]
        y = self._state[1]
        new_x = x
        new_y = y

        p = 0.8 + self.eps
        act = False
        if np.random.random_sample() < p:
            act = True

        if int(u) == 0:
            if y > 0 and act:
                new_y = y - 1
        elif int(u) == 1:
            if y < self.height-1 and act:
                new_y = y + 1
        elif int(u) == 2:
            if x < self.width-1 and act:
                new_x = x + 1
        elif int(u) == 3:
            if x > 0 and act:
                new_x = x - 1

        reward = self.calculateReward(np.array([new_x,new_y]))
        if (new_x,new_y) in self.walls:
            new_x = x
            new_y = y

        new_state = np.array([new_x,new_y])
        if new_x == self.goal[0] and new_y == self.goal[1]:
            #ßprint("GOAL!")
            self._absorbing = True
        else:
            self._absorbing = False

        self._state = new_state

        return new_state, reward, self._absorbing, {}

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            self._state = self.start
        else:
            self._state = state
        return self.get_state()

    def get_state(self):
        return self._state

    def _render(self, mode=None, close=None):
        pass

    def calculateReward(self,state):
        #print(str(state[0])+","+str(state[1]))
        if self.goal[0] == state[0] and self.goal[1] == state[1]:
            return 1
        if state[0] <= 4:
            return (-5+state[0])
        return -(state[0]-4)
