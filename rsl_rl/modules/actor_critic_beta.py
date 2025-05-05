from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta

# Speed up distribution construction by disabling checking
Beta.set_default_validate_args(False)


class ActorCriticBeta(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_alpha_beta=1.0,  # Changed from init_noise_std
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        # Policy network
        self.actor = self.create_mlp(
            num_actor_obs,
            actor_hidden_dims,
            2 * num_actions,  # Output both alpha and beta parameters
            activation
        )
        # Value function network
        self.critic = self.create_mlp(
            num_critic_obs,
            critic_hidden_dims,
            1,
            activation
        )
        print("Actor network: ", self.actor)
        print("Critic network: ", self.critic)
        print("Distribution: Beta")

        # Initialize Beta distribution parameters
        self.initialize_beta_params(init_alpha_beta)
        self.distribution = None

    def create_mlp(self, input_dim, hidden_dims, output_dim, activation):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def initialize_beta_params(self, init_value):
        # Initialize final layer weights for better Beta param initialization
        last_layer = self.actor[-1]
        nn.init.uniform_(last_layer.weight, -init_value, init_value)
        nn.init.constant_(last_layer.bias, init_value)

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # Get alpha and beta from network output
        params = self.actor(observations)
        alpha, beta = params.chunk(2, dim=-1)
        # Apply softplus to ensure alpha and beta are > 1
        alpha = nn.functional.softplus(alpha) + 1.0
        beta = nn.functional.softplus(beta) + 1.0

        # Create Beta distribution
        self.distribution = Beta(alpha, beta)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.rsample()
        return 2.0 * actions - 1.0  # Scale actions to [-1, 1]

    def get_actions_log_prob(self, actions):
        # If you scaled actions, use inverse transform here
        raw_actions = (actions + 1.0) / 2.0
        return self.distribution.log_prob(raw_actions).sum(dim=-1) - torch.log(torch.tensor(2.0))

    def act_inference(self, observations):
        self.update_distribution(observations)
        return self.distribution.mode

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


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
