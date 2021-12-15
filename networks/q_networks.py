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


class DQN(nn.Module):
    def __init__(self, width: int = 84, height: int = 84, num_actions: int = 100):
        super(DQN, self).__init__()

        self.encoder = Encoder()
        self.fc = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x.view(x.size(0), -1))

        return F.softplus(self.head(x))


class DuelingDQN(nn.Module):
    def __init__(self, width: int = 84, height: int = 84, num_actions: int = 100):
        super(DuelingDQN, self).__init__()

        self.encoder = Encoder()

        self.fc_advantages = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_advantages = nn.Linear(64, num_actions)

        self.fc_value = nn.Linear(self.encoder.hidden_dimensions(width, height), 64)
        self.head_value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.encoder(x)

        x_value = self.fc_value(x.view(x.size(0), -1))
        x_value = F.softplus(self.head_value(x_value))

        x_advantages = self.fc_advantages(x.view(x.size(0), -1))
        x_advantages = self.head_advantages(x_advantages)

        return x_value + x_advantages - x_advantages.mean()
