from ray import tune

from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import env.shogi_simple.shogi_pettingzoo_env as shogi_pz
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

from ray.rllib.agents.dqn import APEX_DEFAULT_CONFIG

from models.DqnCNN import TorchMaskedActions

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    def env_creator():
        env = shogi_pz.get_env()
        return env

    env = "shogi"
    register_env(env, lambda config: PettingZooEnv(env_creator()))

    ModelCatalog.register_custom_model("dqn-CNN", TorchMaskedActions)

    config = APEX_DEFAULT_CONFIG
    config["num_gpus"] = 0
    config["num_workers"] = 10
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

