from shogi_pettingzoo_env import *
from tianshou.env.PettingZooEnv import PettingZooEnv

env = get_env()
env.reset()

PettingZooEnv(shogi_pettingzoo_env.get_env())

env.render()
print(np.where(env.observe("player_0")["action_mask"] == 1))
env.step(57)
env.render()
print(env.last())
