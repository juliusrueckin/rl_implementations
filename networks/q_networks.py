import math

import torch
from torch import nn
from torch.nn import functional as F

import constants as const


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(const.FRAMES_STACKED, 32, kernel_size=(4, 4), stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(64)

    def hidden_dimensions(self, width: int = 84, height: int = 84) -> int:
        x = F.relu(self.bn1(self.conv1(torch.rand((1, const.FRAMES_STACKED, width, height)))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet learned exploration.
    Implementation from: https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class DQN(nn.Module):
    def __init__(self, width: int, height: int, num_actions: int, noisy_net: bool = False, noisy_std_init: float = 0.5):
        super(DQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.encoder = Encoder()
        self.fc = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head = nn.Linear(64, num_actions)

        if noisy_net:
            self.fc = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head = NoisyLinear(64, num_actions, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x.view(x.size(0), -1))

        return F.softplus(self.head(x))

    def reset_noisy_layers(self):
        if not self.noisy_net:
            return

        self.fc.reset_noise()
        self.head.reset_noise()


class DuelingDQN(nn.Module):
    def __init__(self, width: int, height: int, num_actions: int, noisy_net: bool = False, noisy_std_init: float = 0.5):
        super(DuelingDQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.encoder = Encoder()

        self.fc_advantages = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_advantages = nn.Linear(64, num_actions)

        if noisy_net:
            self.fc_advantages = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head_advantages = NoisyLinear(64, num_actions, noisy_std_init)

        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_value = nn.Linear(64, 1)

        if noisy_net:
            self.fc_value = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head_value = NoisyLinear(64, 1, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)

        x_value = self.fc_value(x.view(x.size(0), -1))
        x_value = F.softplus(self.head_value(x_value))

        x_advantages = self.fc_advantages(x.view(x.size(0), -1))
        x_advantages = self.head_advantages(x_advantages)

        return x_value + x_advantages - x_advantages.mean()

    def reset_noisy_layers(self):
        if not self.noisy_net:
            return

        self.fc_advantages.reset_noise()
        self.head_advantages.reset_noise()
        self.fc_value.reset_noise()
        self.head_value.reset_noise()
