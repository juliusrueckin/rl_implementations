import os
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.utils.tensorboard import SummaryWriter

import sac_constants as const
from networks.sac_networks import PolicyNet, QNet
from q_learning.replay_buffers import ExperienceReplay
from utils.utils import (
    explained_variance,
    Transition,
    clip_gradients,
    polyak_averaging,
    compute_grad_norm,
    log_network_params,
    ValueStats,
)


class SACWrapper:
    def __init__(
        self, width: int, height: int, num_actions: int, action_limits: torch.tensor, writer: SummaryWriter = None
    ):
        self.value_stats = ValueStats()
        self.num_actions = num_actions
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ExperienceReplay(
            const.REPLAY_BUFFER_LEN,
            const.BATCH_SIZE,
            const.N_STEP_RETURNS,
            const.GAMMA,
        )

        self.policy_net = PolicyNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            action_limits,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)

        self.q_net1 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.target_q_net1 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net1.eval()

        self.q_net2 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.target_q_net2 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.target_q_net2.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.POLICY_LEARNING_RATE)
        self.q_net1_optimizer = torch.optim.Adam(self.q_net1.parameters(), lr=const.Q_VALUE_LEARNING_RATE)
        self.q_net2_optimizer = torch.optim.Adam(self.q_net2.parameters(), lr=const.Q_VALUE_LEARNING_RATE)

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

    def update_target_network(self):
        with torch.no_grad():
            if const.TARGET_UPDATE > 1:
                self.target_q_net1.load_state_dict(self.q_net1.state_dict())
                self.target_q_net2.load_state_dict(self.q_net2.state_dict())
            else:
                polyak_averaging(self.target_q_net1, self.q_net1, const.TAU)
                polyak_averaging(self.target_q_net2, self.q_net2, const.TAU)

    @staticmethod
    def get_policy_loss(log_probs: torch.tensor, q_values: torch.tensor) -> torch.tensor:
        return (const.ENTROPY_COEFF * log_probs - q_values).mean()

    @staticmethod
    def get_q_value_loss(estimated_q_values: torch.tensor, q_value_targets: torch.tensor) -> torch.Tensor:
        q_value_loss_fn = nn.MSELoss(reduction="mean")
        return q_value_loss_fn(estimated_q_values, q_value_targets.detach())

    def prepare_batch_data(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        transitions, _, _ = self.replay_buffer.sample()
        batch = Transition(*zip(*transitions))

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

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def optimize_model(self, steps_done: int):
        for _ in range(const.NUM_GRADIENT_STEPS):
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
            ) = self.prepare_batch_data()

            new_action_batch, log_prob_batch, new_policy = self.policy_net.evaluate(state_batch, reparameterize=True)

            with torch.no_grad():
                new_next_action_batch, new_next_log_prob_batch, _ = self.policy_net.evaluate(
                    next_state_batch, reparameterize=False
                )
                target_next_q_values = torch.min(
                    self.target_q_net1(next_state_batch, new_next_action_batch),
                    self.target_q_net2(next_state_batch, new_next_action_batch),
                )
                target_q_values = reward_batch.squeeze() + (const.GAMMA ** const.N_STEP_RETURNS) * (
                    1 - done_batch.squeeze()
                ) * (target_next_q_values.squeeze() - const.ENTROPY_COEFF * new_next_log_prob_batch.squeeze())
                if const.NORMALIZE_VALUES:
                    target_q_values = self.value_stats.normalize(target_q_values, shift_mean=False)

            estimated_q_values1 = self.q_net1(state_batch, action_batch.unsqueeze(1)).squeeze()
            estimated_q_values2 = self.q_net2(state_batch, action_batch.unsqueeze(1)).squeeze()
            q_value_loss1 = self.get_q_value_loss(estimated_q_values1.float(), target_q_values.float())
            q_value_loss2 = self.get_q_value_loss(estimated_q_values2.float(), target_q_values.float())

            self.q_net1_optimizer.zero_grad()
            q_value_loss1.backward()
            clip_gradients(self.q_net1, const.CLIP_GRAD)
            self.q_net1_optimizer.step()

            self.q_net2_optimizer.zero_grad()
            q_value_loss2.backward()
            clip_gradients(self.q_net2, const.CLIP_GRAD)
            self.q_net2_optimizer.step()

            estimated_new_q_values = torch.min(
                self.q_net1(state_batch, new_action_batch), self.q_net2(state_batch, new_action_batch)
            )
            policy_loss = self.get_policy_loss(log_prob_batch.float(), estimated_new_q_values.float())

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_gradients(self.policy_net, const.CLIP_GRAD)
            self.policy_optimizer.step()

        if self.writer:
            self.track_tensorboard_metrics(
                steps_done,
                policy_loss,
                q_value_loss1,
                q_value_loss2,
                estimated_q_values1,
                target_q_values,
                explained_variance(estimated_q_values1, target_q_values),
                new_policy,
            )

    def track_tensorboard_metrics(
        self,
        steps_done: int,
        policy_loss: float,
        q_value_loss1: float,
        q_value_loss2: float,
        estimated_q_values: torch.tensor,
        target_q_values: torch.tensor,
        explained_var: float,
        policy: Independent,
    ):
        self.writer.add_scalar(f"Training/Policy-Entropy", policy.entropy().mean(), steps_done)
        self.writer.add_scalar(f"Training/Policy-Mean", policy.base_dist.loc.mean(), steps_done)
        self.writer.add_scalar(f"Training/Policy-Std", policy.base_dist.scale.mean(), steps_done)
        self.writer.add_scalar("Training/Policy-Loss", policy_loss, steps_done)

        self.writer.add_scalar("Training/Q-Value1-Loss", q_value_loss1, steps_done)
        self.writer.add_scalar("Training/Q-Value2-Loss", q_value_loss2, steps_done)
        self.writer.add_scalar("Training/Q-Value1-Estimation", estimated_q_values.mean(), steps_done)
        self.writer.add_scalar("Training/Q-Value-Target", target_q_values.mean(), steps_done)
        self.writer.add_scalar("Training/Q-Value-Explained-Variance", explained_var, steps_done)

        self.writer.add_scalar(f"Training/Policy-GradNorm", compute_grad_norm(self.policy_net), steps_done)
        self.writer.add_scalar(f"Training/Q-Value1-GradNorm", compute_grad_norm(self.q_net1), steps_done)
        self.writer.add_scalar(f"Training/Q-Value2-GradNorm", compute_grad_norm(self.q_net2), steps_done)

        if steps_done % 500 == 0:
            log_network_params(self.policy_net, self.writer, steps_done, "Policy-Parameters")
            log_network_params(self.q_net1, self.writer, steps_done, "Q1-Net-Parameters")
            log_network_params(self.q_net2, self.writer, steps_done, "Q2-Net-Parameters")
