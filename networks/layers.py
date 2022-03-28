import math

import torch
import torch.nn.functional as F
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, num_frames: int, num_channels: int):
        super(CNNEncoder, self).__init__()

        self.num_frames = num_frames
        self.conv1 = nn.Conv2d(num_frames, int(num_channels / 2), kernel_size=(4, 4), stride=4)
        self.conv2 = nn.Conv2d(int(num_channels / 2), num_channels, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), stride=1)

    def hidden_dimensions(self, width: int = 84, height: int = 84) -> int:
        x = F.silu(self.conv1(torch.rand((1, self.num_frames, width, height))))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))

        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))

        return x


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, num_hidden_units: int):
        super(MLPEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_hidden_units)
        self.fc2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.fc3 = nn.Linear(num_hidden_units, num_hidden_units)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))

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
