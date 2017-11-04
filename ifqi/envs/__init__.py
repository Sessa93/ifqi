from .acrobot import Acrobot
#from .bicycle import Bicycle
#from .carOnHill import CarOnHill
#from .cartpole import CartPole
#from .gymenv import Gym
#from .invertedPendulum import InvPendulum
#from .lqg1d import LQG1D
#from .swimmer import Swimmer
#from .swingPendulum import SwingPendulum
#from .synthetic import SyntheticToyFS
from .utils import get_space_info
#from .gridworld import GridWorldEnv
#from .atari import Atari
#from .simplegrid import SimpleGrid
from .puddleworld import PuddleWorld
from .puddleworld2 import PuddleWorld2
from .puddleworld3 import PuddleWorld3
from .damV2 import Dam

#__all__ = ['Acrobot', 'Atari', 'Bicycle', 'CarOnHill', 'CartPole', 'GridWorldEnv', 'Gym', 'InvPendulum','LQG1D', 'Swimmer', 'SwingPendulum', 'SyntheticToyFS','SimpleGrid','PuddleWorld','PuddleWorldS']
__all__ = ['PuddleWorld', 'PuddleWorld2','Acrobot', 'PuddleWorld3', 'Dam']
