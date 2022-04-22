import ray
import pickle
import torch
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import ApexTrainer
from shogi_simple.shogi_pettingzoo_env import get_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from ray.rllib.agents.registry import get_agent_class
from copy import deepcopy
import argparse
from pathlib import Path
from shogi_apex_random_policy import TorchMaskedActions


parser = argparse.ArgumentParser(description='Render pretrained policy loaded from checkpoint')
parser.add_argument("checkpoint_path", help="Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`")
parser.add_argument("--play", action='store_true')
parser.set_defaults(play=False)

args = parser.parse_args()

print(args.play)
checkpoint_path = os.path.expanduser(args.checkpoint_path)
params_path = Path(checkpoint_path).parent.parent/"params.pkl"

ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

def env_creator():
    env = get_env()
    return env

register_env("shogi", lambda config: PettingZooEnv(env_creator()))

env = (env_creator())

with open(params_path, "rb") as f:
    config = pickle.load(f)
    # num_workers not needed since we are not training
    del config['num_workers']
    #del config['num_gpus']
    #del config['replay_buffer_shards_colocated_with_driver']
#    if config['exploration_config']:
    del config['exploration_config']

ApexAgent = ApexTrainer(env="shogi", config=config)
ApexAgent.restore(checkpoint_path)

reward_sums = {a:0 for a in env.possible_agents}
env.reset()

for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    obs = observation['observation']
    reward_sums[agent] += reward
    if done:
        action = None
    else:
        if args.play and agent == "player_1":
            print(np.where(observation["action_mask"] == 1)[0])
            action = int(input("Action"))
        else:
            policy = ApexAgent.get_policy("player_0")
            batch_obs = {
                'obs':{
                    'observation': torch.tensor(np.expand_dims(observation['observation'], 0), dtype=torch.float),
                    'action_mask': torch.tensor(np.expand_dims(observation['action_mask'], 0), dtype=torch.float)
                }
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action
            print(action)

    env.step(action)
    env.render()

print("rewards:")
print(reward_sums)
