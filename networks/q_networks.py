from torch import nn
from torch.nn import functional as F

from networks.layers import NoisyLinear, Encoder


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
