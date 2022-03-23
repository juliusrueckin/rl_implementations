from torch import nn
from torch.nn import functional as F

from networks.layers import NoisyLinear, CNNEncoder


class DQN(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        noisy_net: bool = False,
        noisy_std_init: float = 0.5,
        num_atoms: int = 1,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(DQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.encoder = CNNEncoder(num_channels)
        self.fc = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.head = nn.Linear(num_fc_hidden_units, num_actions * num_atoms)

        if noisy_net:
            self.fc = NoisyLinear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units, noisy_std_init)
            self.head = NoisyLinear(num_fc_hidden_units, num_actions * num_atoms, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)

        if self.num_atoms == 1:
            return F.softplus(x)

        x = x.view(x.size(0), self.num_actions, self.num_atoms)
        return F.softmax(x, dim=2)

    def reset_noisy_layers(self):
        if not self.noisy_net:
            return

        self.fc.reset_noise()
        self.head.reset_noise()


class DuelingDQN(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        noisy_net: bool = False,
        noisy_std_init: float = 0.5,
        num_atoms: int = 1,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
    ):
        super(DuelingDQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.encoder = CNNEncoder(num_channels)

        self.fc_advantages = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.head_advantages = nn.Linear(num_fc_hidden_units, num_actions * num_atoms)

        if noisy_net:
            self.fc_advantages = NoisyLinear(
                self.encoder.hidden_dimensions(width, height), num_fc_hidden_units, noisy_std_init
            )
            self.head_advantages = NoisyLinear(num_fc_hidden_units, num_actions * num_atoms, noisy_std_init)

        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), num_fc_hidden_units)
        self.head_value = nn.Linear(num_fc_hidden_units, num_atoms)

        if noisy_net:
            self.fc_value = NoisyLinear(
                self.encoder.hidden_dimensions(width, height), num_fc_hidden_units, noisy_std_init
            )
            self.head_value = NoisyLinear(num_fc_hidden_units, num_atoms, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)

        x_value = F.relu(self.fc_value(x.view(x.size(0), -1)))
        x_value = self.head_value(x_value)

        x_advantages = F.relu(self.fc_advantages(x.view(x.size(0), -1)))
        x_advantages = self.head_advantages(x_advantages)

        if self.num_atoms == 1:
            return F.softplus(x_value) + x_advantages - x_advantages.mean()

        x_value = x_value.view(x_value.size(0), 1, self.num_atoms)
        x_advantages = x_advantages.view(x_advantages.size(0), self.num_actions, self.num_atoms)
        x = x_value + x_advantages - x_advantages.mean(1, keepdim=True)

        return F.softmax(x, dim=2)

    def reset_noisy_layers(self):
        if not self.noisy_net:
            return

        self.fc_advantages.reset_noise()
        self.head_advantages.reset_noise()
        self.fc_value.reset_noise()
        self.head_value.reset_noise()
