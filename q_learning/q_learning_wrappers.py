import os
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import q_learning_constants as const
from networks import get_network
from q_learning.replay_buffers import PrioritizedExperienceReplay
from utils import utils


class DeepQLearningBaseWrapper:
    def __init__(
        self, screen_width: int, screen_height: int, num_actions: int, network_name: str, writer: SummaryWriter = None
    ):
        self.num_actions = num_actions
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = PrioritizedExperienceReplay(
            const.REPLAY_BUFFER_LEN,
            const.BATCH_SIZE,
            const.N_STEP_RETURNS,
            const.ALPHA,
            const.BETA0,
            const.REPLAY_DELAY,
        )

        self.policy_net = get_network(
            network_name,
            screen_width,
            screen_height,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
        ).to(self.device)
        self.target_net = get_network(
            network_name,
            screen_width,
            screen_height,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.LEARNING_RATE)

        self.episode_returns = deque([], maxlen=const.EPISODES_PATIENCE)
        self.max_mean_episode_return = -np.inf

    def episode_terminated(self, episode_return: float, steps_done: int):
        self.writer.add_scalar("Episode/Return", episode_return, steps_done)
        self.episode_returns.append(episode_return)
        running_mean_return = sum(self.episode_returns) / len(self.episode_returns)
        if len(self.episode_returns) >= const.EPISODES_PATIENCE and running_mean_return > self.max_mean_episode_return:
            self.max_mean_episode_return = running_mean_return
            best_model_file_path = os.path.join(const.LOG_DIR, "best_policy_net.pth")
            torch.save(self.policy_net, best_model_file_path)

    @staticmethod
    def get_loss(estimated_q_values: torch.tensor, td_targets: torch.tensor) -> torch.tensor:
        if const.NUM_ATOMS == 1:
            loss_fn = nn.SmoothL1Loss(reduction="none")
            return loss_fn(estimated_q_values, td_targets)

        estimated_q_values = estimated_q_values.clamp(min=1e-5)
        loss = -(td_targets * torch.log(estimated_q_values)).sum(1)
        return loss

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_policy_net_eval(self):
        pass

    def prepare_batch_data(
        self,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        transitions, indices, weights = self.replay_buffer.sample()
        weights = torch.FloatTensor(weights).to(self.device)
        batch = utils.Transition(*zip(*transitions))

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat(batch.reward),
            torch.cat(batch.next_state),
            torch.cat(batch.done),
        )

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
            state_batch.to(self.device),
            action_batch.to(self.device),
            reward_batch.to(self.device),
            next_state_batch.to(self.device),
            done_batch.to(self.device),
        )

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices

    def get_q_value_estimate(self, state_batch: torch.tensor, action_batch: torch.tensor) -> torch.tensor:
        if const.NUM_ATOMS == 1:
            return self.policy_net(state_batch).gather(1, action_batch)

        q_value_dist = self.policy_net(state_batch)
        action_batch = action_batch.unsqueeze(1).expand(action_batch.size(0), 1, const.NUM_ATOMS)
        return q_value_dist.gather(1, action_batch).squeeze(1)

    def get_td_targets(
        self,
        reward_batch: torch.tensor,
        next_state_batch: torch.tensor,
        done_batch: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError("Deep Q-Learning wrapper does not implement 'get_td_targets()' function!")

    def get_distributional_td_targets(
        self, reward_batch: torch.tensor, next_q_value_dists: torch.tensor
    ) -> torch.tensor:
        delta_z = float(const.V_MAX - const.V_MIN) / (const.NUM_ATOMS - 1)

        rewards = reward_batch.unsqueeze(1).expand_as(next_q_value_dists)
        support = (
            torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)
            .unsqueeze(0)
            .expand_as(next_q_value_dists)
            .to(self.device)
        )

        Tz = rewards + np.power(const.GAMMA, const.N_STEP_RETURNS) * support
        Tz = Tz.clamp(min=const.V_MIN, max=const.V_MAX)
        b = (Tz - const.V_MIN) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        lower[(upper > 0) * (lower == upper)] -= 1
        upper[(lower < (const.NUM_ATOMS - 1)) * (lower == upper)] += 1

        proj_td_targets = torch.zeros_like(next_q_value_dists, device=self.device)
        proj_td_targets.scatter_add_(dim=1, index=lower, src=next_q_value_dists * (upper.float() - b))
        proj_td_targets.scatter_add_(dim=1, index=upper, src=next_q_value_dists * (b - lower.float()))

        return proj_td_targets

    def optimization_step(
        self, estimated_q_values: torch.tensor, td_targets: torch.tensor, weights: np.array, indices: np.array
    ):
        td_errors = self.get_loss(estimated_q_values, td_targets)
        weighted_losses = weights * td_errors
        weighted_loss = weighted_losses.mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-const.CLIP_GRAD, const.CLIP_GRAD)

        self.optimizer.step()

        self.replay_buffer.step()
        self.replay_buffer.update(indices, (td_errors + 1e-8).view(-1).data.cpu().numpy())

        self.policy_net.reset_noisy_layers()
        self.target_net.reset_noisy_layers()
        self.update_policy_net_eval()

        return weighted_loss

    def optimize_model(self, steps_done: int = None):
        if len(self.replay_buffer) < const.MIN_START_STEPS:
            return

        if len(self.replay_buffer) == const.MIN_START_STEPS:
            print(f"START OPTIMIZATION")

        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
            weights,
            indices,
        ) = self.prepare_batch_data()

        estimated_q_values = self.get_q_value_estimate(state_batch, action_batch)
        td_targets, next_state_action_values = self.get_td_targets(reward_batch, next_state_batch, done_batch)
        weighted_loss = self.optimization_step(estimated_q_values, td_targets, weights, indices)

        if self.writer:
            self.track_tensorboard_metrics(
                steps_done,
                weighted_loss,
                td_targets if const.NUM_ATOMS == 1 else self.get_expected_values(td_targets),
                estimated_q_values if const.NUM_ATOMS == 1 else self.get_expected_values(estimated_q_values),
                next_state_action_values,
                estimated_q_values[0] if const.NUM_ATOMS > 1 else None,
                td_targets[0] if const.NUM_ATOMS > 1 else None,
            )

    def get_expected_values(self, value_distributions: torch.tensor) -> torch.tensor:
        return (value_distributions * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS).to(self.device)).sum(1)

    def track_tensorboard_metrics(
        self,
        steps_done: int,
        loss: int,
        td_targets: torch.tensor,
        estimated_q_values: torch.tensor,
        next_q_values: torch.tensor,
        estimated_q_value_distribution: torch.tensor = None,
        td_target_distribution: torch.tensor = None,
    ):
        self.writer.add_scalar("Training/Loss", loss, steps_done)
        self.writer.add_scalar("Hyperparam/Beta", self.replay_buffer.beta, steps_done)

        self.writer.add_scalar("Training/TD-Target", td_targets.mean(), steps_done)
        self.writer.add_scalar("Training/TD-Estimation", estimated_q_values.mean(), steps_done)
        self.writer.add_histogram("OnlineNetwork/NextQValues", next_q_values, steps_done)

        if estimated_q_value_distribution is not None and td_target_distribution is not None:
            self.writer.add_histogram("Distributions/Q-Values", estimated_q_value_distribution, steps_done)
            self.writer.add_histogram("Distributions/TD-Targets", td_target_distribution, steps_done)

        total_grad_norm = 0
        for params in self.policy_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/GradientNorm", total_grad_norm, steps_done)

        if steps_done % 500 == 0:
            for tag, params in self.policy_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Parameters/{tag}", params.data.cpu().numpy(), steps_done)


class DeepQLearningWrapper(DeepQLearningBaseWrapper):
    def __init__(
        self, screen_width: int, screen_height: int, num_actions: int, network_name: str, writer: SummaryWriter = None
    ):
        super().__init__(screen_width, screen_height, num_actions, network_name, writer)

    def get_td_targets(
        self,
        reward_batch: torch.tensor,
        next_state_batch: torch.tensor,
        done_batch: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor]:
        if const.NUM_ATOMS == 1:
            next_state_action_values = (1 - done_batch) * self.target_net(next_state_batch).max(1)[0].detach()
            return (
                (reward_batch + np.power(const.GAMMA, const.N_STEP_RETURNS) * next_state_action_values).unsqueeze(1),
                next_state_action_values,
            )

        next_q_value_dists = (1 - done_batch) * self.target_net(next_state_batch).detach()
        next_actions = (next_q_value_dists * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)).sum(2).max(1)[1]
        next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(reward_batch.size(0), 1, const.NUM_ATOMS)
        next_q_value_dists = next_q_value_dists.gather(1, next_actions).squeeze(1)

        projected_td_targets = self.get_distributional_td_targets(reward_batch, next_q_value_dists)

        return projected_td_targets, self.get_expected_values(next_q_value_dists)


class DoubleDeepQLearningWrapper(DeepQLearningBaseWrapper):
    def __init__(
        self, screen_width: int, screen_height: int, num_actions: int, network_name: str, writer: SummaryWriter = None
    ):
        super().__init__(screen_width, screen_height, num_actions, network_name, writer)

        self.policy_net_eval = get_network(
            network_name,
            screen_width,
            screen_height,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
        ).to(self.device)
        self.policy_net_eval.load_state_dict(self.policy_net.state_dict())
        self.policy_net_eval.eval()

    def update_policy_net_eval(self):
        self.policy_net_eval.load_state_dict(self.policy_net.state_dict())

    def get_td_targets(
        self, reward_batch: torch.tensor, next_state_batch: torch.tensor, done_batch: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        if const.NOISY_NETS:
            self.policy_net_eval.reset_noisy_layers()

        if const.NUM_ATOMS == 1:
            next_state_action_indices = self.policy_net_eval(next_state_batch).detach().max(1)[1]
            next_state_action_indices = next_state_action_indices.view(next_state_action_indices.size(0), 1)

            next_state_action_values = (1 - done_batch) * self.target_net(next_state_batch).detach().gather(
                1, next_state_action_indices
            ).view(-1)

            return (
                (reward_batch + np.power(const.GAMMA, const.N_STEP_RETURNS) * next_state_action_values).unsqueeze(1),
                next_state_action_values,
            )

        next_actions = (
            (
                self.policy_net_eval(next_state_batch).detach()
                * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS).to(self.device)
            )
            .sum(2)
            .max(1)[1]
        )
        next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(next_actions.size(0), 1, const.NUM_ATOMS)

        next_q_value_dists = (1 - done_batch.unsqueeze(1).unsqueeze(1)) * self.target_net(
            next_state_batch
        ).detach().gather(1, next_actions)
        non_final_mask = torch.tensor(tuple(map(lambda d: d == 0, done_batch)), device=self.device, dtype=torch.bool)
        next_q_value_dists[~non_final_mask, :, 0] = 1.0
        next_q_value_dists = next_q_value_dists.squeeze(1)

        projected_td_targets = self.get_distributional_td_targets(reward_batch, next_q_value_dists)

        return projected_td_targets, self.get_expected_values(next_q_value_dists)
