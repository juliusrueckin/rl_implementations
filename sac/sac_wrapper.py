import os
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.utils.tensorboard import SummaryWriter

import sac_constants as const
from networks.sac_networks import PolicyNet, QNet, Curl
from q_learning.replay_buffers import ExperienceReplay
from utils.utils import (
    explained_variance,
    Transition,
    TransitionCurl,
    clip_gradients,
    polyak_averaging,
    compute_grad_norm,
    log_network_params,
    ValueStats,
    normalize_values,
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
            num_latent_dims=const.NUM_LATENT_DIMS,
        ).to(self.device)

        self.q_net1 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
            num_latent_dims=const.NUM_LATENT_DIMS,
        ).to(self.device)
        self.target_q_net1 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
            num_latent_dims=const.NUM_LATENT_DIMS,
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
            num_latent_dims=const.NUM_LATENT_DIMS,
        ).to(self.device)
        self.q_net2.encoder.load_state_dict(self.q_net1.encoder.state_dict())
        self.target_q_net2 = QNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
            num_latent_dims=const.NUM_LATENT_DIMS,
        ).to(self.device)
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        self.target_q_net2.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.POLICY_LEARNING_RATE)
        self.q_net1_optimizer = torch.optim.Adam(self.q_net1.parameters(), lr=const.Q_VALUE_LEARNING_RATE)
        self.q_net2_optimizer = torch.optim.Adam(self.q_net2.parameters(), lr=const.Q_VALUE_LEARNING_RATE)

        self.episode_returns = deque([], maxlen=const.EPISODES_PATIENCE)
        self.max_mean_episode_return = -np.inf

        self.log_entropy_coeff = torch.log(torch.ones(1, device=self.device) * const.INIT_ENTROPY_COEFF).requires_grad_(
            True
        )
        self.entropy_coeff_optimizer = torch.optim.Adam([self.log_entropy_coeff], lr=const.ENTROPY_LEARNING_RATE)
        self.target_entropy = -float(num_actions)

        self.curl = Curl(const.NUM_LATENT_DIMS, const.BATCH_SIZE, self.q_net1, self.target_q_net1)
        self.encoder_optimizer = torch.optim.Adam(self.q_net1.encoder.parameters(), lr=const.ENCODER_LEARNING_RATE)
        self.curl_optimizer = torch.optim.Adam(self.curl.parameters(), lr=const.ENCODER_LEARNING_RATE)

    def episode_terminated(self, episode_return: float, steps_done: int):
        self.writer.add_scalar("Episode/Return", episode_return, steps_done)
        self.episode_returns.append(episode_return)
        running_mean_return = sum(self.episode_returns) / len(self.episode_returns)

        if len(self.episode_returns) >= const.EPISODES_PATIENCE and running_mean_return > self.max_mean_episode_return:
            self.max_mean_episode_return = running_mean_return
            best_model_file_path = os.path.join(const.LOG_DIR, "best_policy_net.pth")
            torch.save(self.policy_net, best_model_file_path)

    def update_target_networks(self):
        with torch.no_grad():
            if const.TARGET_UPDATE > 1:
                self.target_q_net1.load_state_dict(self.q_net1.state_dict())
                self.target_q_net2.load_state_dict(self.q_net2.state_dict())
            else:
                polyak_averaging(self.target_q_net1, self.q_net1, const.TAU)
                polyak_averaging(self.target_q_net2, self.q_net2, const.TAU)

            self.policy_net.encoder.load_state_dict(self.q_net1.encoder.state_dict())
            self.q_net2.encoder.load_state_dict(self.q_net1.encoder.state_dict())

    @staticmethod
    def get_policy_loss(log_probs: torch.tensor, q_values: torch.tensor, ent_coeff: torch.tensor) -> torch.tensor:
        return (ent_coeff.detach() * log_probs - q_values).mean()

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

    def prepare_batch_data_curl(
        self,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        transitions, _, _ = self.replay_buffer.sample_curl(const.INPUT_SIZE)
        batch = TransitionCurl(*zip(*transitions))

        (
            state_batch,
            state_anchor_batch,
            state_target_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = (
            torch.cat(batch.state),
            torch.cat(batch.state_anchor),
            torch.cat(batch.state_target),
            torch.cat(batch.action),
            torch.cat(batch.reward),
            torch.cat(batch.next_state),
            torch.cat(batch.done),
        )

        (
            state_batch,
            state_anchor_batch,
            state_target_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = (
            state_batch.to(self.device),
            state_anchor_batch.to(self.device),
            state_target_batch.to(self.device),
            action_batch.to(self.device),
            reward_batch.to(self.device),
            next_state_batch.to(self.device),
            done_batch.to(self.device),
        )

        return (
            state_batch,
            state_anchor_batch,
            state_target_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        )

    def optimize_model(self, steps_done: int):
        for _ in range(const.NUM_GRADIENT_STEPS):
            (
                state_batch,
                state_anchor_batch,
                state_target_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch,
            ) = self.prepare_batch_data_curl()

            new_action_batch, log_prob_batch, new_policy = self.policy_net.evaluate(
                state_batch, reparameterize=True, detach_encoder=True
            )

            entropy_coeff = torch.exp(self.log_entropy_coeff).detach()
            entropy_coeff_loss = -(self.log_entropy_coeff * (log_prob_batch + self.target_entropy).detach()).mean()

            self.entropy_coeff_optimizer.zero_grad()
            entropy_coeff_loss.backward()
            self.entropy_coeff_optimizer.step()

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
                ) * (target_next_q_values.squeeze() - entropy_coeff.detach() * new_next_log_prob_batch.squeeze())
                if const.NORMALIZE_VALUES:
                    target_q_values = normalize_values(target_q_values, shift_mean=False)

            estimated_q_values1 = self.q_net1(state_batch, action_batch.unsqueeze(1), detach_encoder=False).squeeze()
            estimated_q_values2 = self.q_net2(state_batch, action_batch.unsqueeze(1), detach_encoder=False).squeeze()
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
                self.q_net1(state_batch, new_action_batch, detach_encoder=True).squeeze(),
                self.q_net2(state_batch, new_action_batch, detach_encoder=True).squeeze(),
            )
            policy_loss = self.get_policy_loss(log_prob_batch.float(), estimated_new_q_values.float(), entropy_coeff)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_gradients(self.policy_net, const.CLIP_GRAD)
            self.policy_optimizer.step()

            z_anchor = self.curl.encode(state_anchor_batch)
            z_target = self.curl.encode(state_target_batch, target=True)

            logits = self.curl.compute_logits(z_anchor, z_target)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            xentropy_loss_fn = nn.CrossEntropyLoss(reduction="mean")
            encoder_loss = xentropy_loss_fn(logits, labels)

            self.encoder_optimizer.zero_grad()
            self.curl_optimizer.zero_grad()
            encoder_loss.backward()
            clip_gradients(self.curl, const.CLIP_GRAD)
            self.encoder_optimizer.step()
            self.curl_optimizer.step()

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
                entropy_coeff.item(),
                entropy_coeff_loss,
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
        entropy_coeff: float,
        entropy_coeff_loss: float,
    ):
        self.writer.add_scalar(f"Policy/Entropy", policy.entropy().mean(), steps_done)
        self.writer.add_scalar(f"Policy/Entropy-Coefficient", entropy_coeff, steps_done)
        self.writer.add_scalar(f"Policy/Entropy-Coefficient-Loss", entropy_coeff_loss, steps_done)

        self.writer.add_scalar(f"Policy/Mean", policy.base_dist.loc.mean(), steps_done)
        self.writer.add_scalar(f"Policy/Std", policy.base_dist.scale.mean(), steps_done)
        self.writer.add_scalar("Policy/Loss", policy_loss, steps_done)

        self.writer.add_scalar("Q-Value/1-Loss", q_value_loss1, steps_done)
        self.writer.add_scalar("Q-Value/2-Loss", q_value_loss2, steps_done)
        self.writer.add_scalar("Q-Value/1-Estimation", estimated_q_values.mean(), steps_done)
        self.writer.add_scalar("Q-Value/Target", target_q_values.mean(), steps_done)
        self.writer.add_scalar("Q-Value/Explained-Variance", explained_var, steps_done)

        self.writer.add_scalar(f"Policy/GradNorm", compute_grad_norm(self.policy_net), steps_done)
        self.writer.add_scalar(f"Q-Value/1-GradNorm", compute_grad_norm(self.q_net1), steps_done)
        self.writer.add_scalar(f"Q-Value/2-GradNorm", compute_grad_norm(self.q_net2), steps_done)

        if steps_done % 500 == 0:
            log_network_params(self.policy_net, self.writer, steps_done, "Policy-Net")
            log_network_params(self.q_net1, self.writer, steps_done, "Q1-Net")
            log_network_params(self.q_net2, self.writer, steps_done, "Q2-Net")
