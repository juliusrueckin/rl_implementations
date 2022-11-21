import gym
import matplotlib.pyplot as plt

import q_learning_constants as const
from utils import utils

env = gym.make(const.ENV_NAME, render_mode="rgb_array").unwrapped
env.reset()

plt.figure()
plt.imshow(utils.get_pendulum_screen(env, const.INPUT_SIZE).cpu().squeeze(0).squeeze(0).numpy(), cmap="gray")
plt.title("Example extracted screen")
plt.show()
