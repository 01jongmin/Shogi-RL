import ray
import pickle
import torch
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from pettingzoo.classic import tictactoe_v3
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
import numpy as np
import os
from ray.rlib.agents.registry import get_agent_class
from copy import deepcopy
import argparse
from pathlib import Path
from shogi_complete import TorchMaskedActions

# Parse arguments
parser = argparse.ArgumentParser(description='Render pretrained policy loaded from checkpoint')
parser.add_argument("model_0_checkpoint")
parser.add_argument("model_0_agent_name")

parser.add_argument("model_1_checkpoint")
parser.add_argument("model_1_agent_name")

parser.add_argument("--trials", type=int)

args = parser.parse_args()

# Create environment
def env_creator():
    env = tictactoe_v3.env()
    return env

ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

register_env("shogi", lambda config: PettingZooEnv(env_creator()))

env = (env_creator())

model_0_checkpoint = os.path.expanduser(args.model_0_checkpoint)

if checkpoint0_path != 'None':
    params0_path = Path(checkpoint0_path).parent.parent/"params.pkl"

    with open(params0_path, "rb") as f:
        config0 = pickle.load(f)
        del config0['num_workers']
        del config['exploration_config']

    Agent0 = DQNTr(env="shogi", config=config0)
    Agent0.restore(checkpoint0_path)
else:
    Agent0 = None
    args.compete = False

checkpoint1_path = os.path.expanduser(args.checkpoint1_path)

if checkpoint1_path != 'None':
    params1_path = Path(checkpoint1_path).parent.parent/"params.pkl"

    with open(params1_path, "rb") as f:
        config1 = pickle.load(f)
        # num_workers not needed since we are not training
        del config1['num_workers']
        #del config['num_gpus']
        #    if config['exploration_config']:
        #        del config['exploration_config']

    Agent1 = ApexTrainer(env="shogi", config=config1)
    Agent1.restore(checkpoint1_path)
else:
    Agent1 = None
    args.compete = False

# Assign players
players = {}

if args.switch:
    if args.compete:
        players['player_0'] = Agent1
    elif args.random:
        players['player_0'] = 'random'
    elif args.play:
        players['player_0'] = 'human'
    players['player_1'] = Agent0 if Agent0 is not None else players['player_0']
else:
    if args.compete:
        players['player_1'] = Agent1
    elif args.random:
        players['player_1'] = 'random'
    elif args.play:
        players['player_1'] = 'human'
    players['player_0'] = Agent0 if Agent0 is not None else players['player_1']

print(players)

# Play function
def play(observation, player):
    agent = players[player]
    if agent == 'human':
        print(np.where(observation["action_mask"] == 1)[0])
        action = int(input("Action"))
    elif agent == 'random':
        actions = np.where(observation["action_mask"] == 1)[0]
        action = np.random.choice(actions)
    else:
        policy = agent.get_policy(player)
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
        # prob = torch.softmax(logits[0][actions], dim=0, dtype=torch.float)
        #batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
        #single_action = batched_action[0]
        # action = single_action
        # idx = prob.multinomial(num_samples=1)
        action = int(actions[idx])
        #print(actions)
        #print(logits[0][actions])
        #print(prob)
        #print(prob.sum())
        #print(idx)
        #print(action)
        #print(single_action)
    
    return action


results = {'player0_win': 0, 'player1_win' : 0, 'Draw' : 0}

trials = args.trials
for t in range(trials):
    rewards = {a:0 for a in env.possible_agents}
    env.reset()

    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        obs = observation['observation']
        rewards[agent] += reward
        if done:
            action = None
        else:
            action = play(observation, agent)
        env.step(action)
        if trials == 1:
            env.render()

    if rewards['player_0'] == 1:
        results['player0_win'] += 1
    elif rewards['player_0'] == -1:
        results['player1_win'] += 1
    elif rewards['player_0'] == 0:
        results['Draw'] += 1

print('results:', results)
