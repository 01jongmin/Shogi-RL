from ray import tune
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.classic import tictactoe_v3
from gym.spaces import Box
from ray.rllib.models import ModelCatalog
import torch
from torch import nn
import torch.nn.functional as F
from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG

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
                2,
                64,
                2
            ),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(4 * 64, 128)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(128, num_outputs)

    def forward(self, input_dict, state, seq_lens):
        # print(dir(input_dict))
        # print(input_dict["obs"].shape)
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        return self.policy_fn(model_out), state

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

        #TorchFC(orig_obs_space, action_space, action_space.n, model_config, name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        #print(ype(input_dict["obs"]["observation"].data))
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
        env = tictactoe_v3.env()
        return env

    register_env("tictactoe", lambda config: PettingZooEnv(env_creator()))

    ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)
    config = DQN_DEFAULT_CONFIG
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["multiagent"] = {
        "policies": set(["player_1", "player_2"]),
        "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
    }
    config["model"] = { "custom_model": "dqn-CNN" }
    config["env"] = "tictactoe"
    config["framework"] = "torch"
    config["dueling"] = False
    config["double_q"] = True
    config["hiddens"] = []

    tune.run(
        "DQN",
        name="independent-tictactoe",
        stop={"timesteps_total": 20000},
        checkpoint_freq=1,
        config=config,
        resume=False
    )

