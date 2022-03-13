import os
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import ppo_constants as const
from networks.ppo_networks import PolicyNet, ValueNet
from ppo.batch_memories import BatchMemory
from utils.utils import explained_variance, TransitionPPO, schedule_clip_epsilon


class PPOWrapper:
    def __init__(self, width: int, height: int, num_actions: int, writer: SummaryWriter = None):
        self.num_actions = num_actions
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_memory = BatchMemory(const.BATCH_SIZE)

        self.policy_net = PolicyNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS)
        self.policy_net_old = PolicyNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS)
        self.policy_net_old.load_state_dict(self.policy_net.state_dict())
        self.policy_net_old.eval()

        self.value_net = ValueNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS)
        self.value_net_old = ValueNet(width, height, num_actions, num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS)
        self.value_net_old.load_state_dict(self.value_net.state_dict())
        self.value_net_old.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.POLICY_LEARNING_RATE)
        self.policy_scheduler = torch.optim.lr_scheduler.LinearLR(self.policy_optimizer, 1.0, 0.1, const.NUM_EPISODES)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=const.VALUE_LEARNING_RATE)
        self.value_scheduler = torch.optim.lr_scheduler.LinearLR(self.value_optimizer, 1.0, 0.1, const.NUM_EPISODES)

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

    def update_old_networks(self):
        self.policy_net_old.load_state_dict(self.policy_net.state_dict())
        self.value_net_old.load_state_dict(self.value_net.state_dict())

    @staticmethod
    def get_policy_loss(
        actions: torch.tensor,
        policies: Categorical,
        old_policies: Categorical,
        advantage_estimates: torch.tensor,
        clip_epsilon: float,
    ) -> torch.tensor:
        policy_ratios = torch.exp(policies.log_prob(actions) - old_policies.log_prob(actions).detach())
        clipped_policy_ratios = torch.clamp(policy_ratios, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = torch.min(
            policy_ratios * advantage_estimates.detach(), clipped_policy_ratios * advantage_estimates.detach()
        ).mean()

        entropy_loss = policies.entropy().mean()

        return -policy_loss - const.ENTROPY_LOSS_COEFF * entropy_loss

    @staticmethod
    def get_value_loss(estimated_values: torch.tensor, value_targets: torch.tensor) -> torch.Tensor:
        value_loss_fn = nn.SmoothL1Loss(reduction="mean")
        return value_loss_fn(estimated_values, value_targets.detach())

    def prepare_batch_data(
        self, transitions: List[TransitionPPO]
    ) -> Tuple[torch.Tensor, torch.Tensor, Categorical, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = TransitionPPO(*zip(*transitions))
        state_batch, action_batch, policy_batch, reward_batch, done_batch, value_batch, advantage_batch = (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat([dist.probs for dist in batch.policy]),
            torch.cat(batch.reward),
            torch.cat(batch.done),
            torch.cat(batch.value),
            torch.cat(batch.advantage),
        )

        state_batch, action_batch, policy_batch, reward_batch, done_batch, value_batch, advantage_batch = (
            state_batch.to(self.device),
            action_batch.to(self.device),
            Categorical(policy_batch.to(self.device)),
            reward_batch.to(self.device),
            done_batch.to(self.device),
            value_batch.to(self.device),
            advantage_batch.to(self.device),
        )

        return state_batch, action_batch, policy_batch, reward_batch, done_batch, value_batch, advantage_batch

    def optimize_model(self, steps_done: int):
        clip_epsilon = schedule_clip_epsilon(const.CLIP_EPSILON, steps_done, const.NUM_EPISODES)
        batches = self.batch_memory.get()
        for _ in range(const.NUM_EPOCHS):
            for batch in batches:
                (
                    state_batch,
                    action_batch,
                    old_policy_batch,
                    reward_batch,
                    done_batch,
                    old_value_batch,
                    advantage_batch,
                ) = self.prepare_batch_data(batch)

                policy_batch = self.policy_net(state_batch)
                value_batch = self.value_net(state_batch)

                old_value_batch, value_batch, advantage_batch = (
                    old_value_batch.squeeze(),
                    value_batch.squeeze(),
                    advantage_batch.squeeze(),
                )
                value_target_batch = advantage_batch + old_value_batch
                advantage_batch = advantage_batch / advantage_batch.std().clamp(min=1e-8)
                value_target_batch = value_target_batch / value_target_batch.std().clamp(min=1e-8)

                policy_loss = self.get_policy_loss(
                    action_batch, policy_batch, old_policy_batch, advantage_batch, clip_epsilon
                )
                self.policy_optimizer.zero_grad()
                policy_loss.backward()

                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-const.CLIP_GRAD, const.CLIP_GRAD)

                self.policy_optimizer.step()
                self.policy_scheduler.step()

                value_loss = self.get_value_loss(value_batch, value_target_batch)
                self.value_optimizer.zero_grad()
                value_loss.backward()

                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-const.CLIP_GRAD, const.CLIP_GRAD)

                self.value_optimizer.step()
                self.value_scheduler.step()

        if self.writer:
            self.track_tensorboard_metrics(
                steps_done,
                policy_loss,
                value_loss,
                value_target_batch,
                value_batch,
                advantage_batch,
                policy_batch.entropy().mean(),
                self.policy_scheduler.get_last_lr()[0],
                explained_variance(value_batch, value_target_batch),
                clip_epsilon,
            )

    def track_tensorboard_metrics(
        self,
        steps_done: int,
        policy_loss: float,
        value_loss: float,
        value_targets: torch.tensor,
        estimated_values: torch.tensor,
        estimated_advantages: torch.tensor,
        entropy: float,
        learning_rate: float,
        explained_variance: float,
        clip_epsilon: float,
    ):
        self.writer.add_scalar("Hyperparam/Epsilon", clip_epsilon, steps_done)
        self.writer.add_scalar("Hyperparam/Learning-Rate", learning_rate, steps_done)

        self.writer.add_scalar("Training/Explained-Variance", explained_variance, steps_done)
        self.writer.add_scalar("Training/Policy-Entropy", entropy, steps_done)
        self.writer.add_scalar("Training/Policy-Loss", policy_loss, steps_done)
        self.writer.add_scalar("Training/Value-Loss", value_loss, steps_done)
        self.writer.add_scalar("Training/Value-Target", value_targets.mean(), steps_done)
        self.writer.add_scalar("Training/Value-Estimation", estimated_values.mean(), steps_done)
        self.writer.add_scalar("Training/Advantage-Estimation", estimated_advantages.mean(), steps_done)

        total_grad_norm = 0
        for params in self.policy_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/Policy-GradNorm", total_grad_norm, steps_done)

        total_grad_norm = 0
        for params in self.value_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/Value-GradNorm", total_grad_norm, steps_done)

        if steps_done % 500 == 0:
            for tag, params in self.policy_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Policy-Parameters/{tag}", params.data.cpu().numpy(), steps_done)

        if steps_done % 500 == 0:
            for tag, params in self.value_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Value-Parameters/{tag}", params.data.cpu().numpy(), steps_done)
