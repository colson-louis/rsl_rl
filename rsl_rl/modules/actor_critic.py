#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        # Policy
        self.actor = create_mlp(num_actor_obs, actor_hidden_dims, activation, num_actions)
        # add tanh activation to the last layer
        self.actor.add_module("tanh", nn.Tanh())

        # Value function
        self.critic = create_mlp(num_critic_obs, critic_hidden_dims, activation, 1)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print("Distribution: Normal")

        # Action noise
        self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = torch.exp(self.log_std)
        self.distribution = Normal(mean, mean * 0.0 + std+1e-8)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def create_mlp(input_dim, hidden_dims, activation, output_dim=None):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation)

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(activation)

    if output_dim is not None:
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "mish":
        return nn.Mish()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        print("invalid activation function!")
        return None
