import random
from collections import deque
from typing import List, Tuple

import numpy as np

from utils import utils


class ReplayBuffer:
    def __init__(self, buffer_length: int = 10000, batch_size: int = 32):
        self.buffer_length = buffer_length
        self.batch_size = batch_size
        self.buffer = deque([], maxlen=buffer_length)

    def step(self):
        pass

    def push(self, *args):
        raise NotImplementedError("Replay buffer does not implement 'push()' method!")

    def sample(self) -> Tuple[List, np.array, np.array]:
        raise NotImplementedError("Replay buffer does not implement 'sample()' method!")

    def update(self, indices: np.array, priorities: np.array):
        pass

    def __len__(self):
        return len(self.buffer)


class ExperienceReplay(ReplayBuffer):
    def __init__(self, buffer_length: int = 10000, batch_size: int = 32):
        super().__init__(buffer_length, batch_size)

    def push(self, *args):
        self.buffer.append(utils.Transition(*args))

    def sample(self) -> Tuple[List, np.array, np.array]:
        sample_indices = np.random.choice(len(self.buffer), size=self.batch_size)
        sample_transitions = [self.buffer[i] for i in sample_indices]
        return sample_transitions, sample_indices, np.ones(len(self.buffer))

    def __len__(self):
        return len(self.buffer)


class PrioritizedExperienceReplay:
    def __init__(
        self,
        buffer_length: int = 10000,
        batch_size: int = 32,
        alpha: float = 0.75,
        beta0: float = 0.4,
        replay_delay: int = 1000,
    ):
        self.buffer_length = buffer_length
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.priorities = deque([], maxlen=buffer_length)
        self.total_steps = (buffer_length // batch_size) * replay_delay
        self.buffer = deque([], maxlen=buffer_length)

    def step(self):
        self.beta = np.minimum(self.beta + (1 - self.beta0) / self.total_steps, 1)

    def sample(self) -> Tuple[List, np.array, np.array]:
        priorities_tmp = np.array(self.priorities)
        probabilities = priorities_tmp ** self.alpha
        probabilities /= probabilities.sum()

        sample_indices = np.random.choice(len(self.buffer), size=self.batch_size, p=probabilities, replace=False)
        sample_transitions = [self.buffer[i] for i in sample_indices]

        weights = (probabilities[sample_indices] * len(self.buffer)) ** (-self.beta)
        weights = np.array(weights / weights.max(), dtype=np.float32)

        return sample_transitions, sample_indices, weights

    def push(self, *args):
        if len(self.buffer) == 0:
            self.priorities.append(1.0)
        else:
            self.priorities.append(np.max(np.array(self.priorities)))
            priorities_tmp = np.array(self.priorities)
            priorities_tmp /= priorities_tmp.sum()
            self.priorities = deque(priorities_tmp, maxlen=self.buffer_length)

        self.buffer.append(utils.Transition(*args))

    def update(self, indices: np.array, priorities: np.array):
        priorities_tmp = np.array(self.priorities)
        priorities_tmp[indices] = priorities
        priorities_tmp /= priorities_tmp.sum()
        self.priorities = deque(priorities_tmp, maxlen=self.buffer_length)

    def __len__(self):
        return len(self.buffer)
