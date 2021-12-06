import torch

import constants as const
from utils import utils
from networks.q_networks import DQN
from replay_buffers import ExperienceReplay, PrioritizedExperienceReplay
from torch import nn


class DeepQLearningWrapper:
    def __init__(self, screen_width: int, screen_height: int, num_actions: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = ExperienceReplay(const.REPLAY_BUFFER_LEN, const.BATCH_SIZE)

        self.policy_net = DQN(screen_width, screen_height, num_actions).to(self.device)
        self.target_net = DQN(screen_width, screen_height, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.LEARNING_RATE)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.replay_buffer) < const.MIN_START_STEPS:
            return

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(const.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + const.GAMMA * next_state_values

        losses = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        weighted_losses = weights * losses
        weighted_loss = weighted_losses.mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-const.CLIP_GRAD, const.CLIP_GRAD)

        self.optimizer.step()

        self.replay_buffer.step()
        self.replay_buffer.update(indices, (losses + 1e-8).view(-1).data.cpu().numpy())


class DoubleDeepQLearningWrapper:
    def __init__(self, screen_width: int, screen_height: int, num_actions: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = PrioritizedExperienceReplay(
            const.REPLAY_BUFFER_LEN, const.BATCH_SIZE, const.ALPHA, const.BETA0, const.REPLAY_DELAY
        )

        self.policy_net = DQN(screen_width, screen_height, num_actions).to(self.device)
        self.target_net = DQN(screen_width, screen_height, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.LEARNING_RATE)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.replay_buffer) < const.MIN_START_STEPS:
            return

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

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_action_indices = self.policy_net(non_final_next_states).max(1)[1].detach()
            next_state_action_indices = next_state_action_indices.view(next_state_action_indices.size(0), 1)

            next_state_action_values = torch.zeros(const.BATCH_SIZE, device=self.device)
            next_state_action_values[non_final_mask] = (
                self.target_net(non_final_next_states).gather(1, next_state_action_indices).view(-1)
            )
            expected_state_action_values = reward_batch + const.GAMMA * next_state_action_values

        losses = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        weighted_losses = weights * losses
        weighted_loss = weighted_losses.mean()
        self.optimizer.zero_grad()
        weighted_loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-const.CLIP_GRAD, const.CLIP_GRAD)

        self.optimizer.step()

        self.replay_buffer.step()
        self.replay_buffer.update(indices, (losses + 1e-8).view(-1).data.cpu().numpy())
