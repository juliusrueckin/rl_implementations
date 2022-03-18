import random
from collections import deque
from itertools import count

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

import q_learning_constants as const
from q_learning import get_q_learning_wrapper
from utils import utils

writer = SummaryWriter(log_dir=const.LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME)
env.reset()

init_screen = utils.get_cartpole_screen(env)
_, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n
steps_done = 0

deep_q_learning_wrapper = get_q_learning_wrapper(
    const.DOUBLE_Q_LEARNING, screen_width, screen_height, num_actions, const.NETWORK_NAME, writer
)

for i in range(const.NUM_EPISODES):
    env.reset()
    observation = utils.get_cartpole_screen(env)
    state = deque([torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED)
    state.append(observation)
    state_tensor = torch.stack(tuple(state), dim=1)

    episode_return = 0
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

        writer.add_scalar("Hyperparam/Epsilon", utils.schedule_epsilon(steps_done), steps_done)
        action = utils.select_action(state_tensor, steps_done, num_actions, deep_q_learning_wrapper.policy_net, device)
        _, reward, done, _ = env.step(action.item())
        episode_return += reward
        reward = torch.tensor([reward], device=device)
        done_tensor = torch.tensor([int(done)], dtype=torch.int32, device=device)
        steps_done += 1

        next_observation = utils.get_cartpole_screen(env)
        next_state = state.copy()
        next_state.append(next_observation)
        next_state_tensor = torch.stack(tuple(next_state), dim=1)

        deep_q_learning_wrapper.replay_buffer.push(state_tensor, action, next_state_tensor, reward, done_tensor)
        deep_q_learning_wrapper.optimize_model(steps_done)

        if steps_done % const.TARGET_UPDATE == 0:
            print(f"UPDATE TARGET NETWORK after {steps_done} STEPS")
            deep_q_learning_wrapper.update_target_network()

        if done:
            deep_q_learning_wrapper.episode_terminated(episode_return, steps_done)
            break

        state_tensor = next_state_tensor
        state = next_state.copy()

print("Complete")
env.render()
env.close()
