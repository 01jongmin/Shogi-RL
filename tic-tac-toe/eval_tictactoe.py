import ray
import csv
import pickle
import torch
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from pettingzoo.classic import tictactoe_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from copy import deepcopy
import argparse
from pathlib import Path
from independent import TorchMaskedActions

# Parse arguments
parser = argparse.ArgumentParser(description='Render pretrained policy loaded from checkpoint')
parser.add_argument("model_checkpoint")
parser.add_argument("model_agent_name")

parser.add_argument("--trials", type=int)

args = parser.parse_args()

# Create environment
def env_creator():
    env = tictactoe_v3.env()
    return env

ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

register_env("tictactoe", lambda config: PettingZooEnv(env_creator()))

env = (env_creator())

def play(observation, agent, agent_name):
    if agent_name == 'random':
        actions = np.where(observation["action_mask"] == 1)[0]
        action = np.random.choice(actions)
    else:
        policy = agent.get_policy(agent_name)
        batch_obs = {
        'obs':{
                'observation': torch.tensor(np.expand_dims(observation['observation'], 0), dtype=torch.float),
                'action_mask': torch.tensor(np.expand_dims(observation['action_mask'], 0), dtype=torch.float)
            }
        }
        batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
        single_action = batched_action[0]
        action = single_action
    
    return action

model_checkpoint = os.path.expanduser(args.model_checkpoint)
trials = args.trials
agent_name = args.model_agent_name

results_ls = []

for iterartion_id in range(1, 21):
    Agent0 = None

    params_path = Path(model_checkpoint)/"params.pkl"

    with open(params_path, "rb") as f:
        config0 = pickle.load(f)
        del config0['num_workers']
        del config0['exploration_config']

        Agent0 = DQNTrainer(env="tictactoe", config=config0)
        iter_restore = model_checkpoint + "/checkpoint_" + str(iterartion_id).zfill(6) + "/checkpoint-" + str(iterartion_id)
        print(iter_restore)
        Agent0.restore(iter_restore)

    results = {'iteration': iterartion_id, 'player1_win': 0, 'player2_win' : 0, 'Draw' : 0}

    for t in range(trials):
        rewards = {a:0 for a in env.possible_agents}
        env.reset()

        for agent in env.agent_iter():
            if agent == "player_1":
                Agent = Agent0
                agent_name_input = agent_name
            else:
                Agent = None
                agent_name_input = "random"

            observation, reward, done, info = env.last()
            obs = observation['observation']
            rewards[agent] += reward
            if done:
                action = None
            else:
                action = play(observation, Agent, agent_name_input)
            env.step(action)

        if rewards['player_1'] == 1:
            results['player1_win'] += 1
        elif rewards['player_1'] == -1:
            results['player2_win'] += 1
        elif rewards['player_1'] == 0:
            results['Draw'] += 1

    results_ls.append(results)
    print('results:', results)

keys = results_ls[0].keys()

with open('tictactoe_eval.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_ls)

