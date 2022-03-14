import random
from collections import namedtuple, deque
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

import q_learning_constants as const
from networks.q_networks import DQN, DuelingDQN

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
TransitionPPO = namedtuple(
    "TransitionPPO", ("state", "action", "policy", "reward", "done", "value", "advantage", "return_t")
)
transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(const.INPUT_SIZE, interpolation=Image.CUBIC),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ]
)


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

    return transform(screen)


def schedule_epsilon(steps_done: int) -> float:
    if const.NOISY_NETS:
        return 0

    return max(const.EPS_END, const.EPS_START - steps_done * (const.EPS_START - const.EPS_END) / const.EPS_DECAY)


def select_action(
    state: torch.tensor, steps_done: int, num_actions: int, policy_net: Union[DQN, DuelingDQN], device: torch.device
):
    eps_threshold = schedule_epsilon(steps_done)
    if random.random() > eps_threshold and steps_done > const.MIN_START_STEPS:
        with torch.no_grad():
            if const.NUM_ATOMS == 1:
                return policy_net(state.to(device)).max(1)[1].view(1, 1)

            q_values_dist = policy_net(state.to(device)) * torch.linspace(const.V_MIN, const.V_MAX, const.NUM_ATOMS).to(
                device
            )
            return q_values_dist.sum(2).max(1)[1].view(1, 1)

    return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)


def compute_cumulated_return(rewards: Union[List, deque]) -> float:
    return sum([np.power(const.GAMMA, i) * transition.reward for i, transition in enumerate(rewards)])


def schedule_clip_epsilon(base_epsilon: float, steps_done: int, total_steps: int):
    return np.maximum(1.0 - (steps_done / total_steps), 0.1) * base_epsilon


def explained_variance(values: torch.Tensor, value_targets: torch.Tensor) -> float:
    return (1 - (value_targets - values).var()) / value_targets.var()
