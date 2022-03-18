import random
from itertools import count

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

import sac_constants as const
from sac.sac_wrapper import SACWrapper

writer = SummaryWriter(log_dir=const.LOG_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make(const.ENV_NAME, g=9.81)
env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_limits = torch.from_numpy(env.action_space.high).to(device=device).unsqueeze(0)
steps_done = 0

sac_wrapper = SACWrapper(state_dim, action_dim, action_limits, writer=writer)

for i in range(const.NUM_EPISODES):
    physical_state = env.reset()
    env.render(mode="rgb_array")
    physical_state_tensor = torch.tensor(physical_state, device=device).unsqueeze(0)

    episode_return = 0
    no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
    for t in count():
        if t < no_op_steps:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break

            continue

        if t % const.ACTION_REPETITIONS != 0:
            _, reward, done, _ = env.step(action.cpu().numpy())
            env.render(mode="rgb_array")
            episode_return += reward
            if done:
                sac_wrapper.episode_terminated(episode_return, steps_done)
                break

            continue

        if steps_done < const.MIN_START_STEPS:
            action = torch.from_numpy(env.action_space.sample()).to(device)
        else:
            action = sac_wrapper.policy_net.get_action(physical_state_tensor).squeeze(1).detach()

        next_physical_state, reward, done, _ = env.step(action.cpu().numpy())
        env.render(mode="rgb_array")
        next_physical_state_tensor = torch.tensor(next_physical_state, device=device).unsqueeze(0)
        episode_return += reward
        reward = torch.tensor([reward], device=device)
        steps_done += 1

        sac_wrapper.replay_buffer.push(physical_state_tensor, action, next_physical_state_tensor, reward)

        if steps_done >= const.MIN_START_STEPS:
            sac_wrapper.optimize_model(steps_done)

        if steps_done % const.TARGET_UPDATE == 0:
            sac_wrapper.update_target_network()

        if done:
            sac_wrapper.episode_terminated(episode_return, steps_done)
            break

        physical_state_tensor = next_physical_state_tensor

print("Complete")
env.render()
env.close()
