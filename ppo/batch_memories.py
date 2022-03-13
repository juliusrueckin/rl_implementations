from typing import List

import numpy as np
import torch
from torch.distributions import Distribution
import ppo_constants as const

from utils.utils import TransitionPPO


class BatchMemory:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.transitions = []
        self.states = []

    def clear(self):
        self.transitions = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        policy: Distribution,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        self.transitions.append(TransitionPPO(state, action, policy, reward, done, value, torch.tensor([0])))

    def compute_advantages(self):
        for t in range(len(self) - 1):
            advantage_t = 0
            for i in range(t, len(self) - 1):
                discount_factor = (const.GAMMA * const.LAMBDA) ** i
                advantage_t += discount_factor * (
                    self.transitions[i].reward
                    + const.GAMMA * self.transitions[i + 1].value * (1 - self.transitions[i].done)
                    - self.transitions[i].value
                )

                if self.transitions[i].done:
                    break

            self.transitions[t] = self.transitions[t]._replace(advantage=torch.tensor([advantage_t]))

    def get(self) -> List:
        self.compute_advantages()
        batch_start_indices = np.arange(0, len(self), self.batch_size)
        transition_indices = np.arange(0, len(self), dtype=np.int32)
        np.random.shuffle(transition_indices)
        batches_indices = [transition_indices[i : i + self.batch_size] for i in batch_start_indices]

        batches = []
        for batch_indices in batches_indices:
            batches.append([self.transitions[batch_index] for batch_index in batch_indices])

        return batches

    def __len__(self):
        return len(self.transitions)
