import gym
import utils
import constants as const
import matplotlib.pyplot as plt

env = gym.make(const.ENV_NAME).unwrapped
env.reset()

plt.figure()
plt.imshow(utils.get_screen(env).cpu().squeeze(0).squeeze(0).numpy(), cmap="gray")
plt.title("Example extracted screen")
plt.show()
