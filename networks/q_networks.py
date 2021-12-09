import torch
from torch import nn
from torch.nn import functional as F

import constants as const


class DQN(nn.Module):
    def __init__(self, width: int = 84, height: int = 84, num_actions: int = 100):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(const.FRAMES_STACKED, 32, kernel_size=(8, 8), stride=4)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(64, track_running_stats=False)

        self.fc = nn.Linear(self.hidden_dimensions(width, height), 64)
        self.head = nn.Linear(64, num_actions)

    def hidden_dimensions(self, width: int = 84, height: int = 84) -> int:
        x = F.relu(self.bn1(self.conv1(torch.rand((1, const.FRAMES_STACKED, width, height)))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.fc(x.view(x.size(0), -1))
        return self.head(x)
