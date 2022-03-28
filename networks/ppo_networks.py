from typing import Tuple

import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from networks.layers import CNNEncoder


class PPONet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(PPONet, self).__init__()

        self.num_actions = num_actions
        self.encoder = CNNEncoder(num_frames, num_channels)
        self.fc_policy = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.policy_head = nn.Linear(num_fc_hidden_units, num_actions)
        self.value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        x = self.encoder(x)

        x_policy = F.silu(self.fc_policy(x.view(x.size(0), -1)))
        x_value = F.silu(self.fc_value(x.view(x.size(0), -1)))

        x_policy = self.policy_head(x_policy)
        policy = F.softmax(x_policy, dim=1)

        value = self.value_head(x_value)

        return Categorical(policy), value


class PolicyNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(PolicyNet, self).__init__()

        self.num_actions = num_actions
        self.encoder = CNNEncoder(num_frames, num_channels)
        self.flatten = nn.Flatten()
        self.fc_policy = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.policy_head = nn.Linear(num_fc_hidden_units, num_actions)

    def forward(self, x: torch.Tensor) -> Categorical:
        x = self.encoder(x)
        x = F.silu(self.fc_policy(self.flatten(x)))
        x_policy = self.policy_head(x)
        policy = F.softmax(x_policy, dim=1)

        return Categorical(policy)


class ValueNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(ValueNet, self).__init__()

        self.num_actions = num_actions
        self.encoder = CNNEncoder(num_frames, num_channels)
        self.flatten = nn.Flatten()
        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.silu(self.fc_value(self.flatten(x)))
        x = self.value_head(x)

        return x
