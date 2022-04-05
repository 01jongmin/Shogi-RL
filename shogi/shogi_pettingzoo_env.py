import warnings

import numpy as np
from gym import spaces
from .board import Board

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

board_shape = (4, 3)
vectorize = np.vectorize(lambda obj, a, b: obj != None and obj.symbol == a and obj.color == b)

def get_observation(board, agent_idx):
    state = board.get_perspective_state(agent_idx)
    
    stack = np.empty((18, 4, 3), dtype=np.int8)
    stack[0] = np.full(board_shape, agent_idx)
    stack[1] = np.ones(board_shape)
    
    for i, piece_symbol in enumerate(["JA", "JANG", "SANG"]):
        stack[i + 2] = np.full(board_shape, board.prisoners[agent_idx][piece_symbol])
        stack[i + 2 + 3] = np.full(board_shape, board.prisoners[not agent_idx][piece_symbol])
    
    for i, piece_symbol in enumerate(["JA", "HU", "JANG", "SANG", "WANG"]):
        stack[i + 8] = vectorize(state, piece_symbol, agent_idx)
        stack[i + 8 + 5] = vectorize(state, piece_symbol, not agent_idx)

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
                                        'observation': spaces.Box(low=0, high=2, shape=(18, 4, 3), dtype=np.int8),
                                        'action_mask': spaces.Box(low=0, high=1, shape=(12 * 8 + 3 * 9,), dtype=np.int8)
                                  }) for i in self.agents}

        self.rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {'test': 0} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        observation = get_observation(self.board, self.possible_agents.index(agent))
        legal_moves = self.board.legal_moves(self.possible_agents.index(agent)) if agent == self.agent_selection else []

        action_mask = np.zeros(123, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        
        game_over_result = self.board.step(action, current_index)
        self.agent_selection = self._agent_selector.next()
        
        if game_over_result != None:            
            # print(game_over_result)

            for i, name in enumerate(self.agents):
                self.dones[name] = True

                if game_over_result == -1:
                    self.rewards[name] = 0
                else:
                    self.rewards[name] = 1 if i == game_over_result else -1

                self.infos[name] = {'legal_moves': []}
            
        self._accumulate_rewards()

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

    def render(self, mode='human'):
        self.board.render(0)

    def close(self):
        pass
