from typing import List

import numpy as np
import torch
from torch.distributions import Distribution

import ppo_constants as const
from utils.utils import TransitionPPO


class BatchMemory:
    def __init__(self, batch_size: int, num_envs: int):
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.transitions = {env_id: [] for env_id in range(num_envs)}

    def clear(self):
        self.transitions = {env_id: [] for env_id in range(self.num_envs)}

    def add(
        self,
        env_id: int,
        state: torch.Tensor,
        action: torch.Tensor,
        policy: Distribution,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        self.transitions[env_id].append(TransitionPPO(state, action, policy, reward, done, value, None, None))

    def compute_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        for env_id in range(self.num_envs):
            for t in range(len(self.transitions[env_id])):
                advantage_t = 0
                for i in range(t, len(self.transitions[env_id])):
                    discount_factor = (const.GAMMA * const.LAMBDA) ** (i - t)
                    if self.transitions[env_id][i].done:
                        advantage_t += discount_factor * (
                            self.transitions[env_id][i].reward - self.transitions[env_id][i].value
                        )
                        break

                    if i == const.HORIZON - 1:
                        advantage_t += discount_factor * (
                            self.transitions[env_id][i].reward
                            + last_values[env_id] * (1 - last_dones[env_id])
                            - self.transitions[env_id][i].value
                        )
                        break

                    advantage_t += discount_factor * (
                        self.transitions[env_id][i].reward
                        + const.GAMMA * self.transitions[env_id][i + 1].value
                        - self.transitions[env_id][i].value
                    )

                self.transitions[env_id][t] = self.transitions[env_id][t]._replace(
                    advantage=torch.tensor([advantage_t])
                )

    def compute_returns(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        for env_id in range(self.num_envs):
            for t in range(len(self.transitions[env_id])):
                return_t = 0
                for i in range(t, len(self.transitions[env_id])):
                    discount_factor = const.GAMMA ** (i - t)
                    return_t += discount_factor * self.transitions[env_id][i].reward
                    if self.transitions[env_id][i].done:
                        break

                    if i == const.HORIZON - 1:
                        return_t += self.transitions[env_id][i].reward + discount_factor * last_values[env_id] * (
                            1 - last_dones[env_id]
                        )

                self.transitions[env_id][t] = self.transitions[env_id][t]._replace(return_t=torch.tensor([return_t]))

    def get(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> List:
        self.compute_returns(last_values, last_dones)
        self.compute_advantages(last_values, last_dones)
        if const.NORMALIZE_VALUES:
            self.normalize_values()

        batch_start_indices = np.arange(0, len(self), self.batch_size)
        transition_indices = np.arange(0, len(self), dtype=np.int32)
        np.random.shuffle(transition_indices)
        batches_indices = [transition_indices[i : i + self.batch_size] for i in batch_start_indices]

        batches = []
        for batch_indices in batches_indices:
            batches.append([self.concatenated_transitions[batch_index] for batch_index in batch_indices])

        return batches

    def normalize_values(self):
        adv_std = (
            torch.cat(
                [
                    self.transitions[env_id][t_id].advantage
                    for t_id in range(len(self.transitions[0]))
                    for env_id in range(self.num_envs)
                ]
            )
            .squeeze()
            .std()
            .clamp(min=1e-8)
        )
        for env_id in range(self.num_envs):
            for i in range(len(self.transitions[env_id])):
                norm_advantage = self.transitions[env_id][i].advantage / adv_std
                self.transitions[env_id][i] = self.transitions[env_id][i]._replace(
                    advantage=torch.tensor([norm_advantage])
                )

        return_std = (
            torch.cat(
                [
                    self.transitions[env_id][t_id].return_t
                    for t_id in range(len(self.transitions[0]))
                    for env_id in range(self.num_envs)
                ]
            )
            .squeeze()
            .std()
            .clamp(min=1e-8)
        )
        for env_id in range(self.num_envs):
            for i in range(len(self.transitions[env_id])):
                norm_return = self.transitions[env_id][i].return_t / return_std
                self.transitions[env_id][i] = self.transitions[env_id][i]._replace(return_t=torch.tensor([norm_return]))

    @property
    def concatenated_transitions(self) -> List[TransitionPPO]:
        return [
            self.transitions[env_id][t_id]
            for t_id in range(len(self.transitions[0]))
            for env_id in range(self.num_envs)
        ]

    def __len__(self):
        len_total = 0
        for env_id in range(self.num_envs):
            len_total += len(self.transitions[env_id])

        return len_total
