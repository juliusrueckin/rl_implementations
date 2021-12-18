from torch import nn
from torch.nn import functional as F

from networks.layers import NoisyLinear, Encoder


class DQN(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        noisy_net: bool = False,
        noisy_std_init: float = 0.5,
        num_atoms: int = 1,
    ):
        super(DQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.encoder = Encoder()
        self.fc = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head = nn.Linear(64, num_actions * num_atoms)

        if noisy_net:
            self.fc = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head = NoisyLinear(64, num_actions * num_atoms, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x.view(x.size(0), -1))
        x = self.head(x)

        if self.num_atoms:
            return F.softplus(x)

        x = x.view(x.size(0), self.num_actions, self.num_atoms)
        return F.softmax(x.view(-1, self.num_atoms), dim=-1).view(-1, self.num_actions, self.num_atoms)

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
    ):
        super(DuelingDQN, self).__init__()

        self.noisy_net = noisy_net
        self.noisy_std_init = noisy_std_init

        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.encoder = Encoder()

        self.fc_advantages = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_advantages = nn.Linear(64, num_actions * num_atoms)

        if noisy_net:
            self.fc_advantages = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head_advantages = NoisyLinear(64, num_actions * num_atoms, noisy_std_init)

        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_value = nn.Linear(64, num_atoms)

        if noisy_net:
            self.fc_value = NoisyLinear(self.encoder.hidden_dimensions(width, height), 64, noisy_std_init)
            self.head_value = NoisyLinear(64, num_atoms, noisy_std_init)

    def forward(self, x):
        x = self.encoder(x)

        x_value = self.fc_value(x.view(x.size(0), -1))
        x_value = self.head_value(x_value)

        x_advantages = self.fc_advantages(x.view(x.size(0), -1))
        x_advantages = self.head_advantages(x_advantages)

        if self.num_atoms == 1:
            return F.softplus(x_value) + x_advantages - x_advantages.mean()

        x_value = x_value.view(x_value.size(0), 1, self.num_atoms)
        x_advantages = x_advantages.view(x_advantages.size(0), self.num_actions, self.num_atoms)
        x = x_value + x_advantages - x_advantages.mean(1, keepdim=True)

        return F.softmax(x.view(-1, self.num_atoms), dim=-1).view(-1, self.num_actions, self.num_atoms)

    def reset_noisy_layers(self):
        if not self.noisy_net:
            return

        self.fc_advantages.reset_noise()
        self.head_advantages.reset_noise()
        self.fc_value.reset_noise()
        self.head_value.reset_noise()
