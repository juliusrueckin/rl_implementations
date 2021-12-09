from itertools import count

import gym
import torch
import random
import constants as const
from utils import utils
from networks.q_network_wrappers import DoubleDeepQLearningWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME).unwrapped
env.reset()

init_screen = utils.get_screen(env)
_, _, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n

double_dql_wrapper = DoubleDeepQLearningWrapper(screen_width, screen_height, num_actions)

steps_done = 0
episode_durations = []

for i in range(const.NUM_EPISODES):
    env.reset()
    state = utils.get_screen(env)

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

        action = utils.select_action(state, steps_done, num_actions, double_dql_wrapper.policy_net, device)
        steps_done += 1
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        state = utils.get_screen(env)

        next_state = None if done else state

        double_dql_wrapper.replay_buffer.push(state, action, next_state, reward)
        state = next_state
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
