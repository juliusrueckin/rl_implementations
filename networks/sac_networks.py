from typing import Tuple

import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.nn import functional as F

from networks.layers import CNNEncoder, MLPEncoder


class PolicyNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        action_dim: int,
        action_limits: torch.Tensor,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(PolicyNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_limits = action_limits

        self.encoder = CNNEncoder(num_frames, num_channels)
        self.flatten = nn.Flatten()
        self.layer_norm1 = nn.LayerNorm(self.encoder.hidden_dimensions(width, height))
        self.fc_mean = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.layer_norm2 = nn.LayerNorm(num_fc_hidden_units)
        self.fc_log_std = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.layer_norm3 = nn.LayerNorm(num_fc_hidden_units)
        self.mean_head = nn.Linear(num_fc_hidden_units, action_dim)
        self.log_std_head = nn.Linear(num_fc_hidden_units, action_dim)

    def forward(self, x: torch.Tensor) -> Independent:
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.layer_norm1(x)

        x_mean = F.silu(self.layer_norm2(self.fc_mean(x)))
        mean = self.mean_head(x_mean)

        x_log_std = F.silu(self.layer_norm3(self.fc_log_std(x)))
        log_std = self.log_std_head(x_log_std)
        std = torch.exp(torch.clamp(log_std, -20, 2))

        return Independent(Normal(loc=mean, scale=std), reinterpreted_batch_ndims=1)

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.forward(x)
        action = torch.tanh(policy.sample())
        u = self.action_limits * action
        return action.detach(), u.detach()

    def evaluate(self, x: torch.Tensor, reparameterize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Independent]:
        policy = self.forward(x)
        u = policy.rsample() if reparameterize else policy.sample()
        action = torch.tanh(u)
        log_prob = policy.log_prob(u) - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=1)

        return action, log_prob, policy


class QNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        action_dim: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(QNet, self).__init__()

        self.encoder = CNNEncoder(num_frames, num_channels)
        self.flatten = nn.Flatten()
        self.layer_norm1 = nn.LayerNorm(self.encoder.hidden_dimensions(width, height))
        self.fc_q_value = nn.Linear(self.encoder.hidden_dimensions(width, height) + action_dim, num_fc_hidden_units)
        self.layer_norm2 = nn.LayerNorm(num_fc_hidden_units)
        self.q_value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.layer_norm1(self.flatten(x))
        x = torch.cat([x, a], dim=1)
        x = F.silu(self.layer_norm2(self.fc_q_value(x)))
        x = self.q_value_head(x)

        return x
