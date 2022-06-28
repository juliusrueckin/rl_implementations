import os
import random
from collections import deque
from itertools import count
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import ppo_constants as const
from networks.ppo_networks import PolicyNet, ValueNet
from ppo.batch_memories import BatchMemory
from utils import utils
from utils.utils import explained_variance, TransitionPPO, schedule_clip_epsilon, clip_gradients


class PPOWrapper:
    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int,
        writer: SummaryWriter = None,
        policy_net_checkpoint: str = None,
        resume_training_checkpoint: str = None,
    ):
        self.num_actions = num_actions
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_memory = BatchMemory(const.BATCH_SIZE, const.NUM_ENVS)

        self.policy_net = PolicyNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)

        if policy_net_checkpoint is not None:
            self.policy_net.load_state_dict(torch.load(policy_net_checkpoint, map_location=self.device))

        self.policy_net_old = PolicyNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.policy_net_old.load_state_dict(self.policy_net.state_dict())
        self.policy_net_old.eval()
        self.policy_net_old.share_memory()

        self.value_net = ValueNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.value_net_old = ValueNet(
            width,
            height,
            const.FRAMES_STACKED,
            num_actions,
            num_fc_hidden_units=const.NUM_FC_HIDDEN_UNITS,
            num_channels=const.NUM_CHANNELS,
        ).to(self.device)
        self.value_net_old.load_state_dict(self.value_net.state_dict())
        self.value_net_old.eval()
        self.value_net_old.share_memory()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.POLICY_LEARNING_RATE)
        self.policy_scheduler = torch.optim.lr_scheduler.LinearLR(self.policy_optimizer, 1.0, 0.1, const.NUM_EPISODES)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=const.VALUE_LEARNING_RATE)
        self.value_scheduler = torch.optim.lr_scheduler.LinearLR(self.value_optimizer, 1.0, 0.1, const.NUM_EPISODES)

        self.episode_returns = deque([], maxlen=const.EPISODES_PATIENCE)
        self.max_mean_episode_return = -np.inf

        self.finished_episodes = 0
        self.total_steps_done = 0
        self.resume_training_checkpoint = resume_training_checkpoint

        if resume_training_checkpoint is not None:
            self.resume_training()

    def resume_training(self):
        if not os.path.exists(self.resume_training_checkpoint):
            raise ValueError(f"Checkpoint file '{self.resume_training_checkpoint}' does not exist!")

        checkpoint_dir = torch.load(self.resume_training_checkpoint)

        self.policy_net.load_state_dict(checkpoint_dir["policy_net_state_dict"])
        self.policy_net_old.load_state_dict(self.policy_net.state_dict())
        self.policy_optimizer.load_state_dict(checkpoint_dir["policy_optimizer_state_dict"])
        self.policy_scheduler.load_state_dict(checkpoint_dir["policy_scheduler_state_dict"])

        self.value_net.load_state_dict(checkpoint_dir["value_net_state_dict"])
        self.value_net_old.load_state_dict(self.value_net.state_dict())
        self.value_optimizer.load_state_dict(checkpoint_dir["value_optimizer_state_dict"])
        self.value_scheduler.load_state_dict(checkpoint_dir["value_scheduler_state_dict"])

        self.episode_returns = checkpoint_dir["episode_returns"]
        self.max_mean_episode_return = checkpoint_dir["max_mean_episode_return"]
        self.finished_episodes = checkpoint_dir["finished_episodes"]
        self.total_steps_done = checkpoint_dir["total_steps_done"]

        print(
            f"RESUME TRAINING FROM {self.resume_training_checkpoint} CHECKPOINT WITH "
            f"{self.total_steps_done} STEPS AND {self.finished_episodes} EPISODES"
        )

    def save_training(self):
        save_training_path = os.path.join(const.LOG_DIR, "ppo_checkpoint.pth")
        checkpoint_dir = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "policy_scheduler_state_dict": self.policy_scheduler.state_dict(),
            "value_net_state_dict": self.value_net.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "value_scheduler_state_dict": self.value_scheduler.state_dict(),
            "finished_episodes": self.finished_episodes,
            "total_steps_done": self.total_steps_done,
            "episode_returns": self.episode_returns,
            "max_mean_episode_return": self.max_mean_episode_return,
        }
        torch.save(checkpoint_dir, save_training_path)

    def episode_terminated(self, episode_return: float):
        self.finished_episodes += 1
        self.writer.add_scalar("Training/EpisodeReturn", episode_return, self.finished_episodes)
        self.episode_returns.append(episode_return)

        if self.finished_episodes % const.CHECKPOINT_FREQUENCY == 0:
            self.save_training()

        running_mean_return = sum(self.episode_returns) / len(self.episode_returns)
        if len(self.episode_returns) >= const.EPISODES_PATIENCE and running_mean_return > self.max_mean_episode_return:
            self.max_mean_episode_return = running_mean_return
            best_model_file_path = os.path.join(const.LOG_DIR, "best_policy_net.pth")
            torch.save(self.policy_net.state_dict(), best_model_file_path)

    def update_networks(self):
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
        value_loss_fn = nn.MSELoss(reduction="mean")
        return value_loss_fn(estimated_values, value_targets.detach())

    def prepare_batch_data(
        self, transitions: List[TransitionPPO]
    ) -> Tuple[torch.Tensor, torch.Tensor, Categorical, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = TransitionPPO(*zip(*transitions))
        state_batch, action_batch, policy_batch, reward_batch, advantage_batch, return_batch = (
            torch.cat(batch.state),
            torch.cat(batch.action),
            torch.cat([dist.probs for dist in batch.policy]),
            torch.cat(batch.reward),
            torch.cat(batch.advantage),
            torch.cat(batch.return_t),
        )

        state_batch, action_batch, policy_batch, reward_batch, advantage_batch, return_batch = (
            state_batch.to(self.device),
            action_batch.to(self.device),
            Categorical(policy_batch.to(self.device)),
            reward_batch.to(self.device),
            advantage_batch.to(self.device),
            return_batch.to(self.device),
        )

        return state_batch, action_batch, policy_batch, reward_batch, advantage_batch, return_batch

    def optimize_model(self):
        print(f"OPTIMIZE MODEL at STEP {self.total_steps_done}")
        for _ in range(const.NUM_EPOCHS):
            clip_epsilon = schedule_clip_epsilon(const.CLIP_EPSILON, self.total_steps_done, const.NUM_EPISODES)
            batches = self.batch_memory.get(self.value_net, self.device)

            for batch in batches:
                (
                    state_batch,
                    action_batch,
                    old_policy_batch,
                    reward_batch,
                    advantage_batch,
                    value_target_batch,
                ) = self.prepare_batch_data(batch)

                policy_batch = self.policy_net(state_batch)
                value_batch = self.value_net(state_batch)

                value_batch, advantage_batch, value_target_batch = (
                    value_batch.squeeze(),
                    advantage_batch.squeeze(),
                    value_target_batch.squeeze(),
                )

                policy_loss = self.get_policy_loss(
                    action_batch, policy_batch, old_policy_batch, advantage_batch, clip_epsilon
                )
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                clip_gradients(self.policy_net, const.CLIP_GRAD)
                self.policy_optimizer.step()
                self.policy_scheduler.step()

                value_loss = self.get_value_loss(value_batch, value_target_batch)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                clip_gradients(self.value_net, const.CLIP_GRAD)
                self.value_optimizer.step()
                self.value_scheduler.step()

        self.update_networks()
        self.batch_memory.clear()

        if self.writer:
            self.track_tensorboard_metrics(
                policy_loss,
                value_loss,
                value_target_batch,
                value_batch,
                advantage_batch,
                policy_batch.entropy().mean(),
                self.policy_scheduler.get_last_lr()[0],
                explained_variance(value_target_batch, value_batch),
                clip_epsilon,
            )

    def track_tensorboard_metrics(
        self,
        policy_loss: float,
        value_loss: float,
        value_targets: torch.tensor,
        estimated_values: torch.tensor,
        estimated_advantages: torch.tensor,
        entropy: float,
        learning_rate: float,
        explained_var: float,
        clip_epsilon: float,
    ):
        self.writer.add_scalar("Hyperparam/Epsilon", clip_epsilon, self.total_steps_done)
        self.writer.add_scalar("Hyperparam/Learning-Rate", learning_rate, self.total_steps_done)

        self.writer.add_scalar("Training/Explained-Variance", explained_var, self.total_steps_done)
        self.writer.add_scalar("Training/Policy-Entropy", entropy, self.total_steps_done)
        self.writer.add_scalar("Training/Policy-Loss", policy_loss, self.total_steps_done)
        self.writer.add_scalar("Training/Value-Loss", value_loss, self.total_steps_done)
        self.writer.add_scalar("Training/Value-Target", value_targets.mean(), self.total_steps_done)
        self.writer.add_scalar("Training/Value-Estimation", estimated_values.mean(), self.total_steps_done)
        self.writer.add_scalar("Training/Advantage-Estimation", estimated_advantages.mean(), self.total_steps_done)

        total_grad_norm = 0
        for params in self.policy_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/Policy-GradNorm", total_grad_norm, self.total_steps_done)

        total_grad_norm = 0
        for params in self.value_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/Value-GradNorm", total_grad_norm, self.total_steps_done)

        if self.total_steps_done % 1000 == 0:
            for tag, params in self.policy_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(
                        f"Policy-Parameters/{tag}", params.data.cpu().numpy(), self.total_steps_done
                    )

        if self.total_steps_done % 1000 == 0:
            for tag, params in self.value_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(
                        f"Value-Parameters/{tag}", params.data.cpu().numpy(), self.total_steps_done
                    )

    def eval_policy(self, env):
        print(f"EVALUATE PPO POLICY AFTER {self.total_steps_done} STEPS")
        episode_returns = np.zeros(const.EVAL_EPISODE_COUNT)

        for episode in range(const.EVAL_EPISODE_COUNT):
            env.reset()
            observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
            state = deque(
                [torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED
            )
            state.append(observation)
            state_tensor = torch.stack(tuple(state), dim=1)

            episode_return = 0
            no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
            for step in count():
                if step < no_op_steps:
                    action = torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
                    _, _, done, _ = env.step(action.item())
                    if done:
                        break

                    continue

                if step % const.ACTION_REPETITIONS != 0:
                    _, _, done, _ = env.step(action.item())
                    if done:
                        episode_returns[episode] = episode_return
                        break

                    continue

                with torch.no_grad():
                    policy = self.policy_net_old(state_tensor.to(self.device))
                    action = policy.probs.max(1)[1].view(1)

                _, reward, done, _ = env.step(action.item())
                episode_return += reward

                next_observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
                next_state = state.copy()
                next_state.append(next_observation)
                next_state_tensor = torch.stack(tuple(next_state), dim=1)

                if done:
                    episode_returns[episode] = episode_return
                    break

                state_tensor = next_state_tensor
                state = next_state.copy()

        self.writer.add_scalar("Eval/EpisodeReturn", np.mean(episode_returns), self.total_steps_done)
        env.render()
        env.close()
