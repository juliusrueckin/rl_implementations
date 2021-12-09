"""Reference implementation: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
from itertools import count

import gym
import torch

import constants as const
from utils import utils
from networks.q_network_wrappers import DeepQLearningWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME).unwrapped
env.reset()

init_screen = utils.get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n

deep_q_learning_wrapper = DeepQLearningWrapper(screen_width, screen_height, num_actions)

steps_done = 0
episode_durations = []

for i in range(const.NUM_EPISODES):
    env.reset()
    last_screen = utils.get_screen(env)
    current_screen = utils.get_screen(env)
    state = current_screen - last_screen
    for t in count():
        action = utils.select_action(state, steps_done, num_actions, deep_q_learning_wrapper.policy_net, device)
        steps_done += 1
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = utils.get_screen(env)

        next_state = None
        if not done:
            next_state = current_screen - last_screen

        deep_q_learning_wrapper.replay_buffer.push(state, action, next_state, reward)
        state = next_state
        deep_q_learning_wrapper.optimize_model()

        if steps_done % const.TARGET_UPDATE == 0:
            print(f"UPDATE TARGET NETWORK after {steps_done} STEPS")
            deep_q_learning_wrapper.update_target_network()

        if done:
            episode_durations.append(t + 1)
            utils.plot_durations(episode_durations)
            break

print("Complete")
env.render()
env.close()
