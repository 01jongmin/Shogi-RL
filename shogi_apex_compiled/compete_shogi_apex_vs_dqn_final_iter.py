import ray
import csv
import pickle
import torch
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import ApexTrainer, DQNTrainer
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

# Parse arguments
parser = argparse.ArgumentParser(description='Render pretrained policy loaded from checkpoint')
parser.add_argument("checkpoint0_path", help="Path to the checkpoint of player 0. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`")
parser.add_argument("checkpoint0_agent")
parser.add_argument("checkpoint1_path", help="Path to the checkpoint of player 1. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`")
parser.add_argument("checkpoint1_agent")
parser.add_argument("--trials", type=int)
args = parser.parse_args()

# Create environment
def env_creator():
    env = get_env()
    return env


ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

register_env("shogi", lambda config: PettingZooEnv(env_creator()))

env = (env_creator())

checkpoint0_path = os.path.expanduser(args.checkpoint0_path)
checkpoint0_agent = os.path.expanduser(args.checkpoint0_agent)
checkpoint1_path = os.path.expanduser(args.checkpoint1_path)
checkpoint1_agent = os.path.expanduser(args.checkpoint1_agent)

# Play function
def play(observation, agent_model, player):
    policy = agent_model.get_policy(player)
    batch_obs = {
    'obs':{
            'observation': torch.tensor(np.expand_dims(observation['observation'], 0), dtype=torch.float),
            'action_mask': torch.tensor(np.expand_dims(observation['action_mask'], 0), dtype=torch.float)
        }
    }
    logits, _ = policy.model(batch_obs)
    actions = (torch.clamp(logits[0], 0, 100) > 0).nonzero().flatten()
    if actions.shape == torch.Size([0]): # logits are all negative
        actions = (logits[0] > torch.min(logits[0])).nonzero().flatten()
        idx = np.argmax(actions)
    else:
        prob = logits[0][actions] / logits[0][actions].sum()
        idx = prob.multinomial(num_samples=1)
    action = int(actions[idx])
    
    return action



results_ls = []
trials = args.trials




Agent1 = None
apex_iterartion_id = 288

params_path_1 = Path(checkpoint1_path)/"params.pkl"

with open(params_path_1, "rb") as f:
    config1 = pickle.load(f)
    config1['num_workers'] = 0
    del config1['exploration_config']
    del config1['keep_per_episode_custom_metrics']
    del config1['output_config']
    del config1['disable_env_checking']

    Agent1 = ApexTrainer(env="shogi", config=config1)
    iter_restore = checkpoint1_path + "/checkpoint_" + str(apex_iterartion_id).zfill(6) + "/checkpoint-" + str(apex_iterartion_id)
    Agent1.restore(iter_restore)
    
for iterartion_id in range(100, 10001, 100):
    Agent0 = None
    params_path_0 = Path(checkpoint0_path)/"params.pkl"

    with open(params_path_0, "rb") as f:
        config0 = pickle.load(f)
        config0['num_workers'] = 0
        del config0['exploration_config']

        Agent0 = DQNTrainer(env="shogi", config=config0)
        iter_restore = checkpoint0_path + "/checkpoint_" + str(iterartion_id).zfill(6) + "/checkpoint-" + str(iterartion_id)
        Agent0.restore(iter_restore)

    agent_dict = {"player_0": Agent0, "player_1": Agent1}
    agent_name_dict = {"player_0": checkpoint0_agent, "player_1": checkpoint1_agent}

    results = {'player0_win': 0, 'player1_win' : 0, 'Draw' : 0}

    for t in range(trials):
        rewards = {a:0 for a in env.possible_agents}
        env.reset()

        for agent in env.agent_iter():
            agent_playing = agent_dict[agent]
            observation, reward, done, info = env.last()
            obs = observation['observation']
            rewards[agent] += reward
            if done:
                action = None
            else:
                action = play(observation, agent_playing, agent_name_dict[agent])
            env.step(action)

        if rewards['player_0'] == 1:
            results['player0_win'] += 1
        elif rewards['player_0'] == -1:
            results['player1_win'] += 1
        elif rewards['player_0'] == 0:
            results['Draw'] += 1

    results_ls.append(results)
    print('results:', results)


keys = results_ls[0].keys()

with open('dqn_shogi_eval.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_ls)

