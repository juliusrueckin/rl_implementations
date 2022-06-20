from typing import List

import numpy as np
import torch
from torch.distributions import Distribution

import ppo_constants as const
from networks.ppo_networks import ValueNet
from utils.utils import TransitionPPO


class BatchMemory:
    def __init__(self, batch_size: int, num_envs: int):
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.transitions = {env_id: [] for env_id in range(num_envs)}
        self.last_values = {env_id: 0 for env_id in range(num_envs)}

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
        last_value: torch.Tensor,
    ):
        self.last_values[env_id] = last_value
        self.transitions[env_id].append(TransitionPPO(state, action, policy, reward, done, None, None))

    def compute_advantages(self, value_net: ValueNet, device: torch.device):
        for env_id in range(self.num_envs):
            last_advantage = 0
            for t in reversed(range(len(self.transitions[env_id]))):
                if t == len(self.transitions[env_id]) - 1:
                    next_non_terminal = 1.0 - int(self.transitions[env_id][-1].done)
                    next_values = self.last_values[env_id]
                else:
                    next_non_terminal = 1.0 - int(self.transitions[env_id][t + 1].done)
                    with torch.no_grad():
                        next_values = value_net(self.transitions[env_id][t + 1].state.to(device))

                with torch.no_grad():
                    delta = (
                        self.transitions[env_id][t].reward
                        + const.GAMMA * next_non_terminal * next_values
                        - value_net(self.transitions[env_id][t].state.to(device))
                    )
                last_advantage = delta + const.GAMMA * const.LAMBDA * next_non_terminal * last_advantage
                self.transitions[env_id][t] = self.transitions[env_id][t]._replace(advantage=last_advantage)

    def compute_returns(self, value_net: ValueNet, device: torch.device):
        for env_id in range(self.num_envs):
            for t in range(len(self.transitions[env_id])):
                with torch.no_grad():
                    return_t = self.transitions[env_id][t].advantage + value_net(
                        self.transitions[env_id][t].state.to(device)
                    )
                    self.transitions[env_id][t] = self.transitions[env_id][t]._replace(return_t=return_t)

    def get(self, value_net: ValueNet, device: torch.device) -> List:
        self.compute_advantages(value_net, device)
        self.compute_returns(value_net, device)

        if const.NORMALIZE_VALUES:
            self.normalize_values()

        batch_start_indices = np.arange(0, len(self), self.batch_size)
        transition_indices = np.arange(0, len(self), dtype=np.int32)
        np.random.shuffle(transition_indices)
        batches_indices = [transition_indices[i : i + self.batch_size] for i in batch_start_indices]

        batches = []
        transitions_concatenated = self.concatenated_transitions
        for batch_indices in batches_indices:
            batches.append([transitions_concatenated[batch_index] for batch_index in batch_indices])

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
