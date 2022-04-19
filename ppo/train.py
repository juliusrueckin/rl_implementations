import random
from collections import deque
from itertools import count

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

import ppo_constants as const
from ppo.ppo_wrappers import PPOWrapper
from utils import utils

writer = SummaryWriter(log_dir=const.LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME)
env.reset()

init_screen = utils.get_cartpole_screen(env, const.INPUT_SIZE)
_, screen_height, screen_width = init_screen.shape
num_actions = env.action_space.n

ppo_wrapper = PPOWrapper(screen_width, screen_height, num_actions, writer)
steps_done = 0

for episode in range(const.NUM_EPISODES):
    env.reset()
    observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
    state = deque([torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED)
    state.append(observation)
    state_tensor = torch.stack(tuple(state), dim=1)

    episode_return = 0
    no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
    for step in count():
        if step < no_op_steps:
            action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
            _, _, done, _ = env.step(action.item())
            if done:
                break

            continue

        if step % const.ACTION_REPETITIONS != 0:
            _, _, done, _ = env.step(action.item())
            if done:
                ppo_wrapper.episode_terminated(episode_return, steps_done)
                break

            continue

        with torch.no_grad():
            policy = ppo_wrapper.policy_net_old(state_tensor.to(device))
            value = ppo_wrapper.value_net_old(state_tensor.to(device))
            action = policy.sample()

        _, reward, done, _ = env.step(action.item())
        episode_return += reward
        reward = torch.tensor([reward], device=device)
        steps_done += 1

        next_observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
        next_state = state.copy()
        next_state.append(next_observation)
        next_state_tensor = torch.stack(tuple(next_state), dim=1)

        ppo_wrapper.batch_memory.add(state_tensor, action, policy, reward, done, value)
        if len(ppo_wrapper.batch_memory) % const.HORIZON == 0 and len(ppo_wrapper.batch_memory) > 0:
            print(f"OPTIMIZE MODEL at STEP {steps_done}")
            ppo_wrapper.optimize_model(steps_done)
            ppo_wrapper.update_old_networks()
            ppo_wrapper.batch_memory.clear()

        if steps_done % const.EVAL_FREQUENCY == 0 and episode > 0:
            ppo_wrapper.eval_policy(steps_done)

        if done:
            ppo_wrapper.episode_terminated(episode_return, steps_done)
            break

        state_tensor = next_state_tensor
        state = next_state.copy()

print("Complete")
env.render()
env.close()
