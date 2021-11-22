import random
from collections import deque
import utils


class ReplayBuffer:
    def __init__(self, buffer_length: int = 10000):
        self.buffer = deque([], maxlen=buffer_length)

    def push(self, *args):
        self.buffer.append(utils.Transition(*args))

    def sample(self, batch_size: int = 32):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
