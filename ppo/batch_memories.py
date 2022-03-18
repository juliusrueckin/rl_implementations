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
        self.transitions.append(TransitionPPO(state, action, policy, reward, done, value, None, None))

    def compute_advantages(self, last_value: torch.Tensor, last_done: torch.Tensor):
        for t in range(len(self)):
            advantage_t = 0
            for i in range(t, len(self)):
                discount_factor = (const.GAMMA * const.LAMBDA) ** (i - t)
                if self.transitions[i].done:
                    advantage_t += discount_factor * (self.transitions[i].reward - self.transitions[i].value)
                    break

                if i == const.HORIZON - 1:
                    advantage_t += discount_factor * (
                        self.transitions[i].reward + last_value * (1 - last_done) - self.transitions[i].value
                    )
                    break

                advantage_t += discount_factor * (
                    self.transitions[i].reward + const.GAMMA * self.transitions[i + 1].value - self.transitions[i].value
                )

            self.transitions[t] = self.transitions[t]._replace(advantage=torch.tensor([advantage_t]))

    def compute_returns(self, last_value: torch.Tensor, last_done: torch.Tensor):
        for t in range(len(self)):
            return_t = 0
            for i in range(t, len(self)):
                discount_factor = const.GAMMA ** (i - t)
                return_t += discount_factor * self.transitions[i].reward
                if self.transitions[i].done:
                    break

                if i == const.HORIZON - 1:
                    return_t += self.transitions[i].reward + discount_factor * last_value * (1 - last_done)

            self.transitions[t] = self.transitions[t]._replace(return_t=torch.tensor([return_t]))

    def get(self, last_value: torch.Tensor, last_done: torch.Tensor) -> List:
        self.compute_returns(last_value, last_done)
        self.compute_advantages(last_value, last_done)
        if const.NORMALIZE_VALUES:
            self.normalize_values()

        batch_start_indices = np.arange(0, len(self), self.batch_size)
        transition_indices = np.arange(0, len(self), dtype=np.int32)
        np.random.shuffle(transition_indices)
        batches_indices = [transition_indices[i : i + self.batch_size] for i in batch_start_indices]

        batches = []
        for batch_indices in batches_indices:
            batches.append([self.transitions[batch_index] for batch_index in batch_indices])

        return batches

    def normalize_values(self):
        adv_std = torch.cat([t.advantage for t in self.transitions]).squeeze().std().clamp(min=1e-8)
        for i in range(len(self.transitions)):
            norm_advantage = self.transitions[i].advantage / adv_std
            self.transitions[i] = self.transitions[i]._replace(advantage=torch.tensor([norm_advantage]))

        return_std = torch.cat([t.return_t for t in self.transitions]).squeeze().std().clamp(min=1e-8)
        for i in range(len(self.transitions)):
            norm_return = self.transitions[i].return_t / return_std
            self.transitions[i] = self.transitions[i]._replace(return_t=torch.tensor([norm_return]))

    def __len__(self):
        return len(self.transitions)
