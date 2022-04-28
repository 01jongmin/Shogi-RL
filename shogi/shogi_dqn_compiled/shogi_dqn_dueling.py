from ray import tune

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import env.shogi_simple.shogi_pettingzoo_env as shogi_pz
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from ray.rllib.agents.dqn import DEFAULT_CONFIG as DQN_DEFAULT_CONFIG

from models.DqnCNN import TorchMaskedActions

if __name__ == "__main__":
    def env_creator():
        env = shogi_pz.get_env()
        return env

    env = "shogi"
    register_env(env, lambda config: PettingZooEnv(env_creator()))

    ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

    config = DQN_DEFAULT_CONFIG
    config["num_gpus"] = 0
    config["num_workers"] = 5 
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
        "DQN",
        name="dqn shogi",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=50,
        config=config,
        resume=False
    )

