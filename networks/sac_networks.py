import torch
from torch import nn
from torch.nn import functional as F

from networks.layers import Encoder
from torch.distributions.normal import Normal
from typing import Tuple


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        x_mean = F.relu(self.fc_mean(x.view(x.size(0), -1)))
        x_log_std = F.relu(self.fc_log_std(x.view(x.size(0), -1)))

        x_mean = self.mean_head(x_mean)
        x_log_std = self.log_std_head(x_log_std)
        x_log_std = torch.clamp(x_log_std, -20, 1)

        return x_mean, x_log_std

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(x)
        std = log_std.exp()

        epsilon = Normal(0, 1).sample().to(self.device)
        return torch.tanh(mean + epsilon * std)

    def evaluate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std = log_std.exp()

        epsilon = Normal(0, 1).sample().to(self.device)
        action = torch.tanh(mean + epsilon * std)
        log_prob = Normal(mean, std).log_prob(mean + std * epsilon) - torch.log(1 - action.pow(2) + 1e-8)
        return action, log_prob, epsilon, mean, log_std


class ValueNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        num_fc_hidden_units: int = 256,
    ):
        super(ValueNet, self).__init__()

        self.num_actions = num_actions
        self.encoder = Encoder()
        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.relu(self.fc_value(x.view(x.size(0), -1)))
        x = self.value_head(x)

        return x


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
