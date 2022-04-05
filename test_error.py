import ray
import pickle
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from pettingzoo.classic import tictactoe_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import PIL
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from ray.rllib.agents.registry import get_agent_class
from copy import deepcopy
import argparse
from pathlib import Path
from independent import TorchMaskedActions


parser = argparse.ArgumentParser(description='Render pretrained policy loaded from checkpoint')
parser.add_argument("checkpoint_path", help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`")

args = parser.parse_args()

checkpoint_path = os.path.expanduser(args.checkpoint_path)
params_path = Path(checkpoint_path).parent.parent/"params.pkl"

alg_name = "DQN"
ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)
# function that outputs the environment you wish to register.

def env_creator():
    env = tictactoe_v3.env()
    return env

num_cpus = 1

config = deepcopy(get_agent_class(alg_name)._default_config)

register_env("tictactoe", lambda config: PettingZooEnv(env_creator()))

env = (env_creator())
# obs_space = env.observation_space
# print(obs_space)
# act_space = test_env.action_space

with open(params_path, "rb") as f:
    config = pickle.load(f)
    # num_workers not needed since we are not training
    del config['num_workers']
    del config['num_gpus']
    del config['exploration_config']


ray.init(num_cpus=8, num_gpus=0)
DQNAgent = DQNTrainer(env="tictactoe", config=config)
DQNAgent.restore(checkpoint_path)

reward_sums = {a:0 for a in env.possible_agents}
i = 0
env.reset()


for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    obs = observation['observation']
    reward_sums[agent] += reward
    if done:
        action = None
    else:
        #print(DQNAgent.get_policy(agent))
        if agent == "player_1":
            print(observation["action_mask"])
            action = int(input('Please enter an action: '))
        else:
            policy = DQNAgent.get_policy(agent)
            batch_obs = {
                'obs':{
                    'observation': np.expand_dims(observation['observation'].astype(float), 0),
                    'action_mask': np.expand_dims(observation['action_mask'],0)
                }
            }
            #print(batch_obs["obs"]["observation"].shape)
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action
            print(action)
            print(type(action))

    env.step(action)
    i += 1
    #print(env.render(mode="rgb_array"))
    env.render()

print("rewards:")
print(reward_sums)
