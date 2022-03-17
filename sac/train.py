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

init_screen = utils.get_pendulum_screen(env)
_, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.shape[0]
steps_done = 0

sac_wrapper = SACWrapper(screen_width, screen_height, num_actions, writer=writer)

for i in range(const.NUM_EPISODES):
    env.reset()
    observation = utils.get_pendulum_screen(env)
    state = deque([torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED)
    state.append(observation)
    state_tensor = torch.stack(tuple(state), dim=1)

    episode_return = 0
    no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
    for t in count():
        if t < no_op_steps:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break

            continue

        if t % const.ACTION_REPETITIONS != 0:
            _, reward, done, _ = env.step(u.cpu().numpy())
            episode_return += reward
            if done:
                sac_wrapper.episode_terminated(episode_return, steps_done)
                break

            continue

        if steps_done < const.MIN_START_STEPS:
            u = env.action_space.sample()
            u = torch.from_numpy(u).to(device)
            action = torch.tanh(u).to(device)
        else:
            action = sac_wrapper.policy_net.get_action(state_tensor).squeeze(1).detach()
            u = torch.from_numpy(env.action_space.high).to(device) * action

        _, reward, done, _ = env.step(u.cpu().numpy())
        episode_return += reward
        reward = torch.tensor([reward], device=device)
        steps_done += 1

        next_observation = utils.get_pendulum_screen(env)
        next_state_tensor = None
        next_state = state.copy()
        if not done:
            next_state.append(next_observation)
            next_state_tensor = torch.stack(tuple(next_state), dim=1)

        sac_wrapper.replay_buffer.push(state_tensor, action, next_state_tensor, reward)

        if steps_done >= const.MIN_START_STEPS:
            sac_wrapper.optimize_model(steps_done)

        if steps_done % const.TARGET_UPDATE == 0:
            sac_wrapper.update_target_network()

        if done:
            sac_wrapper.episode_terminated(episode_return, steps_done)
            break

        state_tensor = next_state_tensor
        state = next_state.copy()

print("Complete")
env.render()
env.close()
