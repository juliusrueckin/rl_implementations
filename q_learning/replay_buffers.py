from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from utils import utils


class ReplayBuffer:
    def __init__(
        self, buffer_length: int = 10000, batch_size: int = 32, n_steps: int = 3, gamma: float = 0.99, num_envs: int = 1
    ):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.buffer_length = int(buffer_length // num_envs) + 1
        self.batch_size = batch_size
        self.n_step_buffer = {env_id: deque([], maxlen=n_steps) for env_id in range(num_envs)}
        self.buffer = {env_id: deque([], maxlen=self.buffer_length) for env_id in range(num_envs)}
        self.beta = 1
        self.gamma = gamma

    def load_state_dict(self, checkpoint_dir: Dict):
        raise NotImplementedError("Replay buffer does not implement 'load_state_dict()' method!")

    def state_dict(self) -> Dict:
        raise NotImplementedError("Replay buffer does not implement 'state_dict()' method!")

    def step(self):
        pass

    def clear_n_step_buffer(self, env_id: int):
        for trans_id in range(1, len(self.n_step_buffer[env_id])):
            self.add_new_priority(env_id)
            n_step_transition = utils.Transition(
                *(
                    self.n_step_buffer[env_id][trans_id].state,
                    self.n_step_buffer[env_id][trans_id].action,
                    self.n_step_buffer[env_id][trans_id].next_state,
                    utils.compute_cumulated_return(list(self.n_step_buffer[env_id])[trans_id:], self.gamma),
                    self.n_step_buffer[env_id][-1].done,
                )
            )
            self.buffer[env_id].append(n_step_transition)

        self.n_step_buffer[env_id] = deque([], maxlen=self.n_steps)

    def add_new_priority(self, env_id: int):
        pass

    def push(self, env_id: int, *args):
        one_step_transition = utils.Transition(*args)
        self.n_step_buffer[env_id].append(one_step_transition)
        if len(self.n_step_buffer[env_id]) < self.n_steps:
            return

        n_step_transition = utils.Transition(
            *(
                self.n_step_buffer[env_id][0].state,
                self.n_step_buffer[env_id][0].action,
                one_step_transition.next_state,
                utils.compute_cumulated_return(self.n_step_buffer[env_id], self.gamma),
                one_step_transition.done,
            )
        )

        self.add_new_priority(env_id)
        self.buffer[env_id].append(n_step_transition)

        if one_step_transition.done:
            self.clear_n_step_buffer(env_id)

    def sample(self) -> Tuple[List, np.array, np.array]:
        raise NotImplementedError("Replay buffer does not implement 'sample()' method!")

    def sample_curl(self, crop_size: int):
        raise NotImplementedError("Replay buffer does not implement 'sample_curl()' method!")

    def update(self, indices: np.array, priorities: np.array):
        pass

    @property
    def buffer_flattened(self) -> Dict:
        raise NotImplementedError("Replay buffer does not implement 'buffer_flattened()' method!")

    def __len__(self):
        len_total = 0
        for env_id in range(self.num_envs):
            len_total += len(self.buffer[env_id])

        return len_total


class ExperienceReplay(ReplayBuffer):
    def __init__(
        self, buffer_length: int = 10000, batch_size: int = 32, n_steps: int = 3, gamma: float = 0.99, num_envs: int = 1
    ):
        super().__init__(buffer_length, batch_size, n_steps, gamma, num_envs)

    def load_state_dict(self, checkpoint_dir: Dict):
        self.n_step_buffer = checkpoint_dir["n_step_buffer"]
        self.buffer = checkpoint_dir["buffer"]

    def state_dict(self) -> Dict:
        return {
            "n_step_buffer": self.n_step_buffer,
            "buffer": self.buffer,
        }

    def sample(self) -> Tuple[List, np.array, np.array]:
        sample_indices = np.random.choice(list(self.buffer_flattened.keys()), size=self.batch_size)
        sample_transitions = [self.buffer_flattened[idx][2] for idx in sample_indices]
        return sample_transitions, sample_indices, np.ones(self.batch_size)

    def sample_curl(self, crop_size: int) -> Tuple[List, np.array, np.array]:
        sample_indices = np.random.choice(list(self.buffer_flattened.keys()), size=self.batch_size)
        sample_transitions = [self.buffer_flattened[idx][2] for idx in sample_indices]
        for i, transition in enumerate(sample_transitions):
            curl_transition = utils.TransitionCurl(
                *(
                    utils.random_crop(transition.state, crop_size),
                    utils.random_crop(transition.state, crop_size),
                    utils.random_crop(transition.state, crop_size),
                    transition.action,
                    utils.random_crop(transition.next_state, crop_size),
                    transition.reward,
                    transition.done,
                )
            )
            sample_transitions[i] = curl_transition

        return sample_transitions, sample_indices, np.ones(self.batch_size)

    @property
    def buffer_flattened(self) -> Dict:
        flattened_buffer = {}
        idx = 0
        for env_id in range(self.num_envs):
            for t_id in range(len(self.buffer[env_id])):
                flattened_buffer[idx] = (env_id, t_id, self.buffer[env_id][t_id])
                idx += 1

        return flattened_buffer


class PrioritizedExperienceReplay(ReplayBuffer):
    def __init__(
        self,
        buffer_length: int = 10000,
        batch_size: int = 32,
        n_steps: int = 3,
        alpha: float = 0.75,
        beta0: float = 0.4,
        replay_delay: int = 1000,
        gamma: float = 0.99,
        num_envs: int = 1,
    ):
        super().__init__(buffer_length, batch_size, n_steps, gamma, num_envs)

        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.priorities = {env_id: deque([], maxlen=self.buffer_length) for env_id in range(num_envs)}
        self.total_steps = (buffer_length // batch_size) * replay_delay

    def load_state_dict(self, checkpoint_dir: Dict):
        self.beta = checkpoint_dir["beta"]
        self.n_step_buffer = checkpoint_dir["n_step_buffer"]
        self.buffer = checkpoint_dir["buffer"]
        self.priorities = checkpoint_dir["priorities"]

    def state_dict(self) -> Dict:
        return {
            "beta": self.beta,
            "n_step_buffer": self.n_step_buffer,
            "buffer": self.buffer,
            "priorities": self.priorities,
        }

    def step(self):
        self.beta = np.minimum(self.beta + (1 - self.beta0) / self.total_steps, 1)

    def sample(self) -> Tuple[List, np.array, np.array]:
        flattened_buffer = self.buffer_flattened
        priorities_tmp = np.array([entry[3] for _, entry in flattened_buffer.items()])
        probabilities = priorities_tmp ** self.alpha
        probabilities /= probabilities.sum()

        sample_indices = np.random.choice(
            list(flattened_buffer.keys()), size=self.batch_size, p=probabilities, replace=False
        )
        sample_transitions = [flattened_buffer[idx][2] for idx in sample_indices]

        weights = (probabilities[sample_indices] * len(self.buffer)) ** (-self.beta)
        weights = np.array(weights / weights.max(), dtype=np.float32)

        return sample_transitions, sample_indices, weights

    def sample_curl(self, crop_size: int) -> Tuple[List, np.array, np.array]:
        flattened_buffer = self.buffer_flattened
        priorities_tmp = np.array([entry[3] for _, entry in flattened_buffer.items()])
        probabilities = priorities_tmp ** self.alpha
        probabilities /= probabilities.sum()

        sample_indices = np.random.choice(
            list(flattened_buffer.keys()), size=self.batch_size, p=probabilities, replace=False
        )
        sample_transitions = [flattened_buffer[idx][2] for idx in sample_indices]

        for i, transition in enumerate(sample_transitions):
            curl_transition = utils.TransitionCurl(
                *(
                    utils.random_crop(transition.state, crop_size),
                    utils.random_crop(transition.state, crop_size),
                    utils.random_crop(transition.state, crop_size),
                    transition.action,
                    utils.random_crop(transition.next_state, crop_size),
                    transition.reward,
                    transition.done,
                )
            )
            sample_transitions[i] = curl_transition

        weights = (probabilities[sample_indices] * len(self.buffer)) ** (-self.beta)
        weights = np.array(weights / weights.max(), dtype=np.float32)

        return sample_transitions, sample_indices, weights

    def add_new_priority(self, env_id: int):
        if len(self.buffer[env_id]) == 0:
            self.priorities[env_id].append(1.0)
        else:
            self.priorities[env_id].append(np.max(np.array([entry[3] for _, entry in self.buffer_flattened.items()])))
            priorities_tmp = np.array(self.priorities[env_id])
            self.priorities[env_id] = deque(priorities_tmp, maxlen=self.buffer_length)

    def update(self, indices: np.array, priorities: np.array):
        buf_flattened = self.buffer_flattened
        for i, flattened_idx in enumerate(indices):
            env_id, t_id, _, _ = buf_flattened[flattened_idx]
            self.priorities[env_id][t_id] = np.abs(priorities[i])

    @property
    def buffer_flattened(self) -> Dict:
        flattened_buffer = {}
        idx = 0
        for env_id in range(self.num_envs):
            for t_id in range(len(self.buffer[env_id])):
                flattened_buffer[idx] = (env_id, t_id, self.buffer[env_id][t_id], self.priorities[env_id][t_id])
                idx += 1

        return flattened_buffer
