import ray
from ray.rllib.agents.registry import get_agent_class
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from shogi.shogi_pettingzoo_env import get_env 
from gym.spaces import Box
from ray.rllib.models import ModelCatalog
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.agents.dqn import APEX_DEFAULT_CONFIG

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                2,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                2,
                padding=1
            ),
            nn.Flatten(),
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model(input_dict["obs"]), state

class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)


        obs_len = obs_space.shape[0]-action_space.n

        orig_obs_space = Box(shape=(obs_len,), low=obs_space.low[:obs_len], high=obs_space.high[:obs_len])
        self.action_embed_model = CNNModelV2(orig_obs_space, action_space, action_space.n, model_config, name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]['observation']
        })
        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    def env_creator():
        env = get_env()
        return env

    env = "shogi"
    register_env(env, lambda config: PettingZooEnv(env_creator()))

    ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

    # Max memory to be set to 90GB
    ray.init(object_store_memory=10**9)

    config = APEX_DEFAULT_CONFIG
    config["num_gpus"] = 0
    config["num_workers"] = 110
    config["multiagent"]  = {
        "policies": set(["player_0", "player_1"]),
        "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
    }
    config["model"] = { "custom_model": "dqn-CNN" }
    config["env"] = env
    config["framework"] = "torch"
    config["dueling"] = False
    config["double_q"] = True
    config["hiddens"] = []

    tune.run(
        "APEX",
        name="apex shogi",
        stop={"timesteps_total": 100000000},
        checkpoint_freq=1,
        config=config,
        resume=False
    )

