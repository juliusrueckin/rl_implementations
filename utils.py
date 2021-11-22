import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from collections import namedtuple

import constants as const
from dqn.q_networks import DQN


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
resize = T.Compose([T.ToPILImage(), T.Resize(const.INPUT_SIZE, interpolation=Image.CUBIC), T.ToTensor()])


def get_cart_location(env, screen_width: int) -> int:
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env) -> torch.Tensor:
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)

    slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)


def select_action(state, steps_done: int, num_actions: int, policy_net: DQN, device: torch.device):
    eps_threshold = max(
        const.EPS_END, const.EPS_START - steps_done * (const.EPS_START - const.EPS_END) / const.EPS_DECAY
    )
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

    return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)


def plot_durations(episode_durations: List):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
