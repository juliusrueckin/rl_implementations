import os
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn
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
)


class SACWrapper:
    def __init__(self, width: int, height: int, num_actions: int, writer: SummaryWriter = None):
        self.num_actions = num_actions
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ExperienceReplay(
            const.REPLAY_BUFFER_LEN,
            const.BATCH_SIZE,
            const.N_STEP_RETURNS,
        )

        self.policy_net = PolicyNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS)

        self.q_net1 = QNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS).to(self.device)
        self.target_q_net1 = QNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS).to(
            self.device
        )
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net1.eval()

        self.q_net2 = QNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS).to(self.device)
        self.target_q_net2 = QNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS).to(
            self.device
        )
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
                polyak_averaging(self.target_q_net1, self.q_net1)
                polyak_averaging(self.target_q_net2, self.q_net2)

    @staticmethod
    def get_policy_loss(log_probs: torch.tensor, q_values: torch.tensor) -> torch.tensor:
        return -(q_values.detach() - const.ENTROPY_COEFF * log_probs).mean()

    @staticmethod
    def get_q_value_loss(estimated_q_values: torch.tensor, q_value_targets: torch.tensor) -> torch.Tensor:
        q_value_loss_fn = nn.SmoothL1Loss(reduction="mean")
        return q_value_loss_fn(estimated_q_values, q_value_targets.detach())

    def prepare_batch_data(
        self,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        transitions, indices, weights = self.replay_buffer.sample()
        weights = torch.FloatTensor(weights).to(self.device)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch, action_batch, reward_batch = (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat(batch.reward),
        )

        state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states = (
            state_batch.to(self.device),
            action_batch.to(self.device),
            reward_batch.to(self.device),
            non_final_mask.to(self.device),
            non_final_next_states.to(self.device),
        )

        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states, weights, indices

    def optimize_model(self, steps_done: int):
        for _ in range(const.NUM_GRADIENT_STEPS):
            (
                state_batch,
                action_batch,
                reward_batch,
                non_final_mask,
                non_final_next_states,
                weights,
                indices,
            ) = self.prepare_batch_data()

            with torch.no_grad():
                new_next_action_batch, new_next_log_prob_batch = self.policy_net.evaluate(non_final_next_states)
                target_q_values = reward_batch.squeeze()
                target_next_q_values = torch.min(
                    self.q_net1(non_final_next_states, new_next_action_batch),
                    self.q_net2(non_final_next_states, new_next_action_batch),
                )
                target_q_values[non_final_mask] += (
                    const.GAMMA * target_next_q_values.squeeze()
                    - const.ENTROPY_COEFF * new_next_log_prob_batch.squeeze()
                )

            estimated_q_values1 = self.q_net1(state_batch, action_batch.unsqueeze(1)).squeeze()
            estimated_q_values2 = self.q_net2(state_batch, action_batch.unsqueeze(1)).squeeze()

            q_value_loss1 = self.get_q_value_loss(estimated_q_values1, target_q_values)
            self.q_net1_optimizer.zero_grad()
            q_value_loss1.backward()
            clip_gradients(self.q_net1)
            self.q_net1_optimizer.step()

            q_value_loss2 = self.get_q_value_loss(estimated_q_values2, target_q_values)
            self.q_net2_optimizer.zero_grad()
            q_value_loss2.backward()
            clip_gradients(self.q_net2)
            self.q_net2_optimizer.step()

            new_action_batch, log_prob_batch = self.policy_net.evaluate(state_batch)
            estimated_new_q_values = torch.min(
                self.q_net1(state_batch, new_action_batch), self.q_net2(state_batch, new_action_batch)
            )
            policy_loss = self.get_policy_loss(log_prob_batch, estimated_new_q_values)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_gradients(self.policy_net)
            self.policy_optimizer.step()

        if self.writer:
            self.track_tensorboard_metrics(
                steps_done,
                policy_loss,
                estimated_q_values1,
                target_q_values,
                explained_variance(estimated_q_values1, target_q_values),
            )

    def track_tensorboard_metrics(
        self,
        steps_done: int,
        policy_loss: float,
        estimated_q_values: torch.tensor,
        target_q_values: torch.tensor,
        explained_var: float,
    ):
        self.writer.add_scalar("Training/Q-Value-Explained-Variance", explained_var, steps_done)
        self.writer.add_scalar("Training/Policy-Loss", policy_loss, steps_done)
        self.writer.add_scalar("Training/Q-Value-Estimation", estimated_q_values.mean(), steps_done)
        self.writer.add_scalar("Training/Q-Value-Target", target_q_values.mean(), steps_done)

        self.writer.add_scalar(f"Training/Policy-GradNorm", compute_grad_norm(self.policy_net), steps_done)
        self.writer.add_scalar(f"Training/Q-Value1-GradNorm", compute_grad_norm(self.q_net1), steps_done)
        self.writer.add_scalar(f"Training/Q-Value2-GradNorm", compute_grad_norm(self.q_net2), steps_done)

        if steps_done % 500 == 0:
            log_network_params(self.policy_net, self.writer, steps_done, "Policy-Parameters")
            log_network_params(self.q_net1, self.writer, steps_done, "Q1-Net-Parameters")
            log_network_params(self.q_net2, self.writer, steps_done, "Q2-Net-Parameters")
