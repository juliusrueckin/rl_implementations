import os
import random
from collections import deque
from itertools import count
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
        self,
        screen_width: int,
        screen_height: int,
        num_actions: int,
        network_name: str,
        writer: SummaryWriter = None,
        policy_net_checkpoint: str = None,
        resume_training_checkpoint: str = None,
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
            const.GAMMA,
            const.NUM_ENVS,
        )

        self.policy_net = get_network(
            network_name,
            screen_width,
            screen_height,
            const.FRAMES_STACKED,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
            const.NUM_CHANNELS,
        ).to(self.device)
        self.policy_net.share_memory()

        if policy_net_checkpoint is not None:
            self.policy_net.load_state_dict(torch.load(policy_net_checkpoint, map_location=self.device))

        self.policy_net_eval = None

        self.target_net = get_network(
            network_name,
            screen_width,
            screen_height,
            const.FRAMES_STACKED,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
            const.NUM_CHANNELS,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=const.LEARNING_RATE)

        self.episode_returns = deque([], maxlen=const.EPISODES_PATIENCE)
        self.max_mean_episode_return = -np.inf

        self.finished_episodes = 0
        self.total_steps_done = 0
        self.resume_training_checkpoint = resume_training_checkpoint

    def resume_training(self):
        if not os.path.exists(self.resume_training_checkpoint):
            raise ValueError(f"Checkpoint file '{self.resume_training_checkpoint}' does not exist!")

        checkpoint_dir = torch.load(self.resume_training_checkpoint)

        self.replay_buffer.load_state_dict(checkpoint_dir["replay_buffer_state_dict"])
        self.policy_net.load_state_dict(checkpoint_dir["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint_dir["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint_dir["optimizer_state_dict"])
        if self.policy_net_eval is not None:
            self.policy_net_eval.load_state_dict(checkpoint_dir["policy_net_eval_state_dict"])

        self.episode_returns = checkpoint_dir["episode_returns"]
        self.max_mean_episode_return = checkpoint_dir["max_mean_episode_return"]
        self.finished_episodes = checkpoint_dir["finished_episodes"]
        self.total_steps_done = checkpoint_dir["total_steps_done"]

        print(
            f"RESUME TRAINING FROM {self.resume_training_checkpoint} CHECKPOINT WITH "
            f"{self.total_steps_done} STEPS AND {self.finished_episodes} EPISODES"
        )

    def save_training(self):
        save_training_path = os.path.join(const.LOG_DIR, "q_learning_checkpoint.pth")
        checkpoint_dir = {
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "policy_net_state_dict": self.policy_net.state_dict(),
            "policy_net_eval_state_dict": None if self.policy_net_eval is None else self.policy_net_eval.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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

    @staticmethod
    def get_loss(estimated_q_values: torch.tensor, td_targets: torch.tensor) -> torch.tensor:
        if const.NUM_ATOMS == 1:
            loss_fn = nn.SmoothL1Loss(reduction="none")
            return loss_fn(estimated_q_values, td_targets)

        estimated_q_values = estimated_q_values.clamp(min=1e-5)
        loss = -(td_targets * torch.log(estimated_q_values)).sum(1)
        return loss

    def update_target_network(self):
        with torch.no_grad():
            if const.TARGET_UPDATE > 1:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                utils.polyak_averaging(self.target_net, self.policy_net, const.TAU)

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
        utils.clip_gradients(self.policy_net, const.CLIP_GRAD)
        self.optimizer.step()

        self.replay_buffer.step()
        self.replay_buffer.update(indices, (td_errors + 1e-8).view(-1).data.cpu().numpy())

        self.policy_net.reset_noisy_layers()
        self.target_net.reset_noisy_layers()
        self.update_policy_net_eval()

        return weighted_loss

    def optimize_model(self):
        for _ in range(const.NUM_GRADIENT_STEPS):
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

        if self.total_steps_done % const.TARGET_UPDATE == 0:
            self.update_target_network()

        if self.writer:
            self.track_tensorboard_metrics(
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
        loss: int,
        td_targets: torch.tensor,
        estimated_q_values: torch.tensor,
        next_q_values: torch.tensor,
        estimated_q_value_distribution: torch.tensor = None,
        td_target_distribution: torch.tensor = None,
    ):
        self.writer.add_scalar(
            "Hyperparam/Epsilon",
            utils.schedule_epsilon(
                self.total_steps_done, const.NOISY_NETS, const.EPS_START, const.EPS_END, const.EPS_DECAY
            ),
            self.total_steps_done,
        )

        self.writer.add_scalar("Training/Loss", loss, self.total_steps_done)
        self.writer.add_scalar("Hyperparam/Beta", self.replay_buffer.beta, self.total_steps_done)

        self.writer.add_scalar("Training/TD-Target", td_targets.mean(), self.total_steps_done)
        self.writer.add_scalar("Training/TD-Estimation", estimated_q_values.mean(), self.total_steps_done)
        self.writer.add_histogram("OnlineNetwork/NextQValues", next_q_values, self.total_steps_done)

        if estimated_q_value_distribution is not None and td_target_distribution is not None:
            self.writer.add_histogram("Distributions/Q-Values", estimated_q_value_distribution, self.total_steps_done)
            self.writer.add_histogram("Distributions/TD-Targets", td_target_distribution, self.total_steps_done)

        total_grad_norm = 0
        for params in self.policy_net.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.writer.add_scalar(f"Training/GradientNorm", total_grad_norm, self.total_steps_done)

        if self.total_steps_done % 1000 == 0:
            for tag, params in self.policy_net.named_parameters():
                if params.grad is not None:
                    self.writer.add_histogram(f"Parameters/{tag}", params.data.cpu().numpy(), self.total_steps_done)

    def eval_policy(self, env):
        print(f"EVALUATE Q-LEARNING POLICY AFTER {self.total_steps_done} STEPS")
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
                        break

                    continue

                action = utils.select_action(
                    state_tensor,
                    self.total_steps_done,
                    self.num_actions,
                    self.policy_net,
                    self.device,
                    const.NOISY_NETS,
                    const.EPS_START,
                    const.EPS_END,
                    const.EPS_DECAY,
                    const.MIN_START_STEPS,
                    const.NUM_ATOMS,
                    const.V_MIN,
                    const.V_MAX,
                    eval_mode=True,
                )
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


class DeepQLearningWrapper(DeepQLearningBaseWrapper):
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        num_actions: int,
        network_name: str,
        writer: SummaryWriter = None,
        policy_net_checkpoint: str = None,
        resume_training_checkpoint: str = None,
    ):
        super().__init__(
            screen_width,
            screen_height,
            num_actions,
            network_name,
            writer,
            policy_net_checkpoint,
            resume_training_checkpoint,
        )

        if resume_training_checkpoint is not None:
            self.resume_training()

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
        self,
        screen_width: int,
        screen_height: int,
        num_actions: int,
        network_name: str,
        writer: SummaryWriter = None,
        policy_net_checkpoint: str = None,
        resume_training_checkpoint: str = None,
    ):
        super().__init__(
            screen_width,
            screen_height,
            num_actions,
            network_name,
            writer,
            policy_net_checkpoint,
            resume_training_checkpoint,
        )

        self.policy_net_eval = get_network(
            network_name,
            screen_width,
            screen_height,
            const.FRAMES_STACKED,
            num_actions,
            const.NOISY_NETS,
            const.NOISY_SIGMA_INIT,
            const.NUM_ATOMS,
            const.NUM_FC_HIDDEN_UNITS,
            const.NUM_CHANNELS,
        ).to(self.device)
        self.policy_net_eval.load_state_dict(self.policy_net.state_dict())
        self.policy_net_eval.eval()

        if resume_training_checkpoint is not None:
            self.resume_training()

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
