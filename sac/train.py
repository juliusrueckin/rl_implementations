import random
from collections import deque
from itertools import count

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

import sac_constants as const
from sac.sac_wrapper import SACWrapper
from utils import utils

writer = SummaryWriter(log_dir=const.LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME, g=9.81)
env.reset()

init_screen = utils.get_pendulum_screen(env, const.IMAGE_SIZE)
_, screen_height, screen_width = utils.random_crop(init_screen, const.INPUT_SIZE).shape
action_dim = env.action_space.shape[0]
action_limits = torch.from_numpy(env.action_space.high).to(device=device)
steps_done = 0
sac_wrapper = SACWrapper(screen_width, screen_height, action_dim, action_limits, writer=writer)

for episode in range(const.NUM_EPISODES):
    env.reset()
    observation = utils.get_pendulum_screen(env, const.IMAGE_SIZE)
    state = deque([torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED)
    state.append(observation)
    state_tensor = torch.stack(tuple(state), dim=1)

    episode_return = 0
    no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
    for step in count():
        if step < no_op_steps:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break

            continue

        if step % const.ACTION_REPETITIONS != 0:
            _, reward, done, _ = env.step(u.cpu().numpy())
            episode_return += reward
            if done:
                sac_wrapper.episode_terminated(episode_return, steps_done)
                break

            continue

        if steps_done < const.MIN_START_STEPS:
            u = torch.from_numpy(env.action_space.sample()).to(device)
            action = torch.tanh(u)
        else:
            action, u = sac_wrapper.policy_net.get_action(
                utils.center_crop(state_tensor.to(device), const.INPUT_SIZE), eval_mode=False
            )
            action, u = action.squeeze(1), u.squeeze(1)

        _, reward, done, _ = env.step(u.cpu().numpy())
        env.render(mode="rgb_array")
        episode_return += reward
        reward = torch.tensor([reward], device=device)
        done_tensor = torch.tensor([int(done)], dtype=torch.int32, device=device)
        steps_done += 1

        next_observation = utils.get_pendulum_screen(env, const.IMAGE_SIZE)
        next_state = state.copy()
        next_state.append(next_observation)
        next_state_tensor = torch.stack(tuple(next_state), dim=1)
        sac_wrapper.replay_buffer.push(state_tensor, action, next_state_tensor, reward, done_tensor)

        if steps_done >= const.MIN_START_STEPS and steps_done % const.OPTIMIZATION_UPDATE == 0:
            sac_wrapper.optimize_model(steps_done)

        if steps_done % const.EVAL_FREQUENCY == 0 and episode > 0:
            sac_wrapper.eval_policy(steps_done)

        if done:
            sac_wrapper.episode_terminated(episode_return, steps_done)
            break

        state_tensor = next_state_tensor
        state = next_state.copy()

print("Complete")
env.render()
env.close()
