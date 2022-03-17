from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.nn import functional as F

from networks.layers import Encoder


class PolicyNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
    ):
        super(PolicyNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions

        self.encoder = Encoder()
        self.fc_mean = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.fc_log_std = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.mean_head = nn.Linear(num_fc_hidden_units, num_actions)
        self.log_std_head = nn.Linear(num_fc_hidden_units, num_actions)

    def forward(self, x: torch.Tensor) -> Independent:
        x = self.encoder(x)

        x_mean = F.relu(self.fc_mean(x.view(x.size(0), -1)))
        x_log_std = F.relu(self.fc_log_std(x.view(x.size(0), -1)))

        mean = self.mean_head(x_mean)
        log_std = self.log_std_head(x_log_std)
        std = torch.exp(torch.clamp(log_std, -20, 2))

        return Independent(Normal(loc=mean, scale=std), reinterpreted_batch_ndims=1)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        policy = self.forward(x)
        return torch.tanh(policy.sample())

    def evaluate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Independent]:
        policy = self.forward(x)
        u = policy.rsample()
        action = torch.tanh(u)
        log_prob = policy.log_prob(u) - torch.sum(2 * (np.log(2) - u - F.softplus(-2 * u)), dim=1)
        return action, log_prob, policy


class QNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
    ):
        super(QNet, self).__init__()

        self.num_actions = num_actions
        self.encoder = Encoder()
        self.fc_q_value = nn.Linear(self.encoder.hidden_dimensions(width, height) + num_actions, num_fc_hidden_units)
        self.q_value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.cat([x.view(x.size(0), -1), a], dim=1)
        x = F.relu(self.fc_q_value(x))
        x = self.q_value_head(x)

        return x
