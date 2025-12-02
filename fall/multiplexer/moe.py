import torch
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm
from typing import List
import torch.nn.functional as F
from gymnasium.spaces import Discrete, Box

class MultiplexerNetwork(torch.nn.Module):

    def __init__(self,experts: List[BaseAlgorithm], network_size:List = [32,32],device: str = "cuda"):

        super().__init__()

        self.experts = experts

        action_space = experts[0].action_space    

        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        elif isinstance(action_space, Box):
            self.n_actions = action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(action_space)}")

        self.device = device

        self.net = nn.Sequential(
            nn.Linear(len(self.experts)*self.n_actions, network_size[0]),
            nn.ReLU(),
            *(
                module
                for i in range(1,len(network_size) - 1)
                for module in (
                    nn.Linear(network_size[i], network_size[i+1]),
                    nn.ReLU()
                )
            ),
            nn.Linear(network_size[-1],self.n_actions),
            nn.Softmax(dim=-1)
        ).to(self.device)
    
    def forward(self,obs):

        with torch.inference_mode():
            experts_out = [
                expert.policy.get_distribution(expert.policy.obs_to_tensor(obs)[0]
                    ).distribution.probs
                for expert in self.experts
            ]

        experts_out = torch.cat(experts_out, dim=1).to(self.device)

        out = self.net(experts_out)

        return out


if __name__ == "__main__":
    from stable_baselines3.common.env_util import make_atari_env , make_vec_env
    import gymnasium as gym
    from stable_baselines3 import PPO
    import ale_py
    gym.register_envs(ale_py)

    # env = make_vec_env("PongNoFrameskip-v4",n_envs=10)
    env = make_atari_env("PongNoFrameskip-v4",n_envs=2)
    model = PPO("CnnPolicy", env, n_steps=64, batch_size=64, verbose=0, device = "cpu")
    

    model.learn(100)

    experts = [model]*3

    moe = MultiplexerNetwork(experts,network_size=[16,16,16])

    print(moe.net)

    obs = env.reset()

    out = moe.forward(obs)




