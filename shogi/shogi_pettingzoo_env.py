import warnings
import collections 

import numpy as np
from gym import spaces
from .board import Board

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

board_shape = (4, 3)
vectorize = np.vectorize(lambda obj, a, b: obj != None and obj.symbol == a and obj.color == b)
collections.deque(maxlen=6)

swap_indexer = np.zeros(120, dtype=np.int8)
for i in range(120):
    q, r = divmod(i, 10)
    if r >= 5:
        swap_indexer[q * 10 + r - 5] = i
    else:
        swap_indexer[q * 10 + r + 5] = i

def get_observation(board):
    state = board.state
    
    stack = np.empty((10, 4, 3), dtype=np.int8)
    
    for i, piece_symbol in enumerate(["JA", "HU", "JANG", "SANG", "WANG"]):
        stack[i] = vectorize(state, piece_symbol, 0)
        stack[i + 5] = vectorize(state, piece_symbol, 1) 

    return stack
    
def get_env():
    env = raw_env()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "animal_shogi",
        "is_parallelizable": False
    }

    def __init__(self):
        super().__init__()
        self.board = Board()

        self.agents = [f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(12 * 8 + 3 * 9) for i in self.agents}
        self.observation_spaces = {i: spaces.Dict({
                                        'observation': spaces.Box(low=0, high=2, shape=(10 * 12 + 6 + 2, 4, 3), dtype=np.int8),
                                        'action_mask': spaces.Box(low=0, high=1, shape=(12 * 8 + 3 * 9,), dtype=np.int8)
                                  }) for i in self.agents}

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {'test': 0} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None

        self.board_history = np.zeros((10 * 12, 4, 3), dtype=np.int8)
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        agent_idx = self.agents.index(agent)

        stack = np.empty((128, 4, 3), dtype=np.int8)

        stack[0] = np.full(board_shape, agent_idx)
        stack[1] = np.ones(board_shape)

        for i, piece_symbol in enumerate(["JA", "JANG", "SANG"]):
            stack[i + 2] = np.full(board_shape, self.board.prisoners[agent_idx][piece_symbol])
            stack[i + 2 + 3] = np.full(board_shape, self.board.prisoners[not agent_idx][piece_symbol])

        if agent_idx == 0:
            stack[8:, :, :] = self.board_history
        else:
            rotated = np.rot90(self.board_history, 2, axes=(1, 2))
            stack[8:, :, :] = rotated[swap_indexer]

        legal_moves = self.board.legal_moves(self.possible_agents.index(agent)) if agent == self.agent_selection else []

        action_mask = np.zeros(123, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': stack, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        
        game_over_result = self.board.step(action, current_index)

        self.board_history = np.vstack((get_observation(self.board), self.board_history[:-10, :, :]))

        if (np.equal(self.board_history[0:40], self.board_history[40:80]).all() and 
                np.equal(self.board_history[40:80], self.board_history[80:120]).all()):
                game_over_result = -1

        if game_over_result != None:            
            for i, name in enumerate(self.agents):
                self.dones[name] = True

                if game_over_result == -1:
                    self.rewards[name] = 0
                else:
                    self.rewards[name] = 1 if i == game_over_result else -1

                self.infos[name] = {'legal_moves': []}
            
        self._accumulate_rewards()

        self.agent_selection = self._agent_selector.next()

    def reset(self):
        self.has_reset = True

        self.agents = self.possible_agents[:]

        self.board = Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        observation = get_observation(self.board)
        
        self.board_history[:10, :, :] = observation

    def render(self, mode='human'):
        self.board.render(0)

    def close(self):
        pass
