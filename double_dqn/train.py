import random
from collections import deque
from itertools import count

import gym
import torch

import constants as const
from networks.q_network_wrappers import DoubleDeepQLearningWrapper
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME).unwrapped
env.reset()

init_screen = utils.get_screen(env)
_, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n

double_dql_wrapper = DoubleDeepQLearningWrapper(screen_width, screen_height, num_actions)

steps_done = 0
episode_durations = []

for i in range(const.NUM_EPISODES):
    env.reset()
    observation = utils.get_screen(env)
    state = deque([torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED)
    state.append(observation)
    state_tensor = torch.stack(tuple(state), dim=1)

    no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
    for t in count():
        if t < no_op_steps:
            action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
            _, _, done, _ = env.step(action.item())
            if done:
                break

            continue

        if t % const.ACTION_REPETITIONS != 0:
            _, _, done, _ = env.step(action.item())
            if done:
                break

            continue

        action = utils.select_action(state_tensor, steps_done, num_actions, double_dql_wrapper.policy_net, device)
        steps_done += 1
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        next_observation = utils.get_screen(env)
        next_state_tensor = None
        if not done:
            next_state = state.copy()
            next_state.append(next_observation)
            next_state_tensor = torch.stack(tuple(next_state), dim=1)

        double_dql_wrapper.replay_buffer.push(state_tensor, action, next_state_tensor, reward)
        double_dql_wrapper.optimize_model()

        if steps_done % const.TARGET_UPDATE == 0:
            print(f"UPDATE TARGET NETWORK after {steps_done} STEPS")
            double_dql_wrapper.update_target_network()

        if done:
            episode_durations.append(t + 1)
            if i % 50 == 0:
                utils.plot_durations(episode_durations)
            break

print("Complete")
env.render()
env.close()
