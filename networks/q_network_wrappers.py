from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import constants as const
from networks import get_network
from replay_buffers import PrioritizedExperienceReplay
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
        ).to(self.device)
        self.target_net = get_network(
            network_name,
            screen_width,
            screen_height,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_fn = self.get_loss_fn()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.LEARNING_RATE)

    @staticmethod
    def get_loss_fn():
        if const.NUM_ATOMS == 1:
            return nn.SmoothL1Loss(reduction="none")

        return nn.CrossEntropyLoss(reduction="none")

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
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch, action_batch, reward_batch = (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat(batch.reward),
        )

        state_batch, action_batch, reward_batch, non_final_mask = (
            state_batch.to(self.device),
            action_batch.to(self.device),
            reward_batch.to(self.device),
            non_final_mask.to(self.device),
        )

        return state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states, weights, indices

    def get_q_value_estimate(self, state_batch: torch.tensor, action_batch: torch.tensor) -> torch.tensor:
        if const.NUM_ATOMS == 1:
            return self.policy_net(state_batch).gather(1, action_batch)

        q_value_dist = self.policy_net(state_batch)
        action_batch = action_batch.unsqueeze(1).expand(action_batch.size(0), 1, const.NUM_ATOMS)

        return q_value_dist.gather(1, action_batch).squeeze(1)

    def get_td_targets(
        self, reward_batch: torch.tensor, non_final_next_states: torch.tensor, non_final_mask: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError("Deep Q-Learning wrapper does not implement 'get_td_targets()' function!")

    def optimization_step(
        self, estimated_q_values: torch.tensor, td_targets: torch.tensor, weights: np.array, indices: np.array
    ):
        td_errors = self.loss_fn(estimated_q_values, td_targets)
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
            non_final_mask,
            non_final_next_states,
            weights,
            indices,
        ) = self.prepare_batch_data()

        estimated_q_values = self.get_q_value_estimate(state_batch, action_batch)
        td_targets, next_state_action_values = self.get_td_targets(reward_batch, non_final_next_states, non_final_mask)
        weighted_loss = self.optimization_step(estimated_q_values, td_targets, weights, indices)

        if self.writer:
            self.track_tensorboard_metrics(
                steps_done,
                weighted_loss,
                td_targets,
                estimated_q_values,
                next_state_action_values,
            )

    def track_tensorboard_metrics(
        self,
        steps_done: int,
        loss: int,
        td_targets: torch.tensor,
        estimated_q_values: torch.tensor,
        next_q_values: torch.tensor,
    ):
        self.writer.add_scalar("Training/Loss", loss, steps_done)
        self.writer.add_scalar("Hyperparam/Beta", self.replay_buffer.beta, steps_done)

        self.writer.add_scalar("Training/TD-Target", td_targets.mean(), steps_done)
        self.writer.add_scalar("Training/TD-Estimation", estimated_q_values.mean(), steps_done)
        self.writer.add_histogram("OnlineNetwork/NextQValues", next_q_values, steps_done)

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
        self, reward_batch: torch.tensor, non_final_next_states: torch.tensor, non_final_mask: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        if const.NUM_ATOMS == 1:
            next_state_action_values = torch.zeros(reward_batch.size(0), device=self.device)
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            return (
                (reward_batch + np.power(const.GAMMA, const.N_STEP_RETURNS) * next_state_action_values).unsqueeze(1),
                next_state_action_values,
            )

        delta_z = float(const.V_MAX - const.V_MIN) / (const.NUM_ATOMS - 1)
        support = torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)

        next_state_action_values = torch.zeros(
            (reward_batch.size(0), self.num_actions, const.NUM_ATOMS), device=self.device
        )
        next_state_action_values[non_final_mask, :] = self.target_net(non_final_next_states).data.cpu() * support

        next_actions = next_state_action_values.sum(2).max(1)[1]
        next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(reward_batch.size(0), 1, const.NUM_ATOMS)
        next_state_action_values = next_state_action_values.gather(1, next_actions).squeeze(1)

        rewards = reward_batch.unsqueeze(1).expand_as(next_state_action_values)
        support = support.unsqueeze(0).expand_as(next_state_action_values)
        Tz = rewards + np.power(const.GAMMA, const.N_STEP_RETURNS) * support
        Tz = Tz.clamp(min=const.V_MIN, max=const.V_MAX)

        b = (Tz - const.V_MIN) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()
        offset = (
            torch.linspace(0, (reward_batch.size(0) - 1) * const.NUM_ATOMS, reward_batch.size(0))
            .long()
            .unsqueeze(1)
            .expand(reward_batch.size(0), const.NUM_ATOMS)
        )

        proj_next_state_action_values = torch.zeros(next_state_action_values.size())
        proj_next_state_action_values.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_state_action_values * (upper.float() - b)).view(-1)
        )
        proj_next_state_action_values.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_state_action_values * (b - lower.float())).view(-1)
        )
        expected_q_values = (
            proj_next_state_action_values * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)
        ).sum(1)

        return proj_next_state_action_values, expected_q_values


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
        ).to(self.device)
        self.policy_net_eval.load_state_dict(self.policy_net.state_dict())
        self.policy_net_eval.eval()

    def update_policy_net_eval(self):
        self.policy_net_eval.load_state_dict(self.policy_net.state_dict())

    def get_td_targets(
        self, reward_batch: torch.tensor, non_final_next_states: torch.tensor, non_final_mask: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        with torch.no_grad():
            if const.NOISY_NETS:
                self.policy_net_eval.reset_noisy_layers()

            if const.NUM_ATOMS == 1:
                next_state_action_indices = self.policy_net_eval(non_final_next_states).max(1)[1].detach()
                next_state_action_indices = next_state_action_indices.view(next_state_action_indices.size(0), 1)

                next_state_action_values = torch.zeros(reward_batch.size(0), device=self.device)
                next_state_action_values[non_final_mask] = (
                    self.target_net(non_final_next_states).gather(1, next_state_action_indices).view(-1)
                )

                return (
                    (reward_batch + np.power(const.GAMMA, const.N_STEP_RETURNS) * next_state_action_values).unsqueeze(
                        1
                    ),
                    next_state_action_values,
                )

            delta_z = float(const.V_MAX - const.V_MIN) / (const.NUM_ATOMS - 1)
            support = torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)

            next_actions = self.policy_net_eval(non_final_next_states).sum(2).max(1)[1].detach()
            next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(next_actions.size(0), 1, const.NUM_ATOMS)

            next_state_action_values = torch.zeros((reward_batch.size(0), 1, const.NUM_ATOMS), device=self.device)
            next_state_action_values[non_final_mask] = (
                self.target_net(non_final_next_states).data.cpu() * support
            ).gather(1, next_actions)
            next_state_action_values = next_state_action_values.squeeze(1)

            rewards = reward_batch.unsqueeze(1).expand_as(next_state_action_values)
            support = support.unsqueeze(0).expand_as(next_state_action_values)
            Tz = rewards + np.power(const.GAMMA, const.N_STEP_RETURNS) * support
            Tz = Tz.clamp(min=const.V_MIN, max=const.V_MAX)

            b = (Tz - const.V_MIN) / delta_z
            lower = b.floor().long()
            upper = b.ceil().long()
            offset = (
                torch.linspace(0, (reward_batch.size(0) - 1) * const.NUM_ATOMS, reward_batch.size(0))
                .long()
                .unsqueeze(1)
                .expand(reward_batch.size(0), const.NUM_ATOMS)
            )

            proj_next_state_action_values = torch.zeros(next_state_action_values.size())
            proj_next_state_action_values.view(-1).index_add_(
                0, (lower + offset).view(-1), (next_state_action_values * (upper.float() - b)).view(-1)
            )
            proj_next_state_action_values.view(-1).index_add_(
                0, (upper + offset).view(-1), (next_state_action_values * (b - lower.float())).view(-1)
            )
            expected_q_values = (
                proj_next_state_action_values * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS)
            ).sum(1)

            return proj_next_state_action_values, expected_q_values
