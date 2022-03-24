import random
from collections import namedtuple, deque
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from networks.q_networks import DQN, DuelingDQN

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))
TransitionPPO = namedtuple(
    "TransitionPPO", ("state", "action", "policy", "reward", "done", "value", "advantage", "return_t")
)


def get_cart_location(env, screen_width: int) -> int:
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_cartpole_screen(env, input_size: int) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(input_size, interpolation=Image.CUBIC),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ]
    )
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


def get_pendulum_screen(env, input_size: int) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(input_size, interpolation=Image.CUBIC),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ]
    )
    screen = env.render(mode="rgb_array").transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[
        :, int(screen_width * 0.2) : int(screen_width * 0.8), int(screen_height * 0.2) : int(screen_height * 0.8)
    ]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    return transform(screen)


def schedule_epsilon(steps_done: int, noisy_net: bool, eps_start: float, eps_end: float, eps_decay: float) -> float:
    if noisy_net:
        return 0

    return max(eps_end, eps_start - steps_done * (eps_start - eps_end) / eps_decay)


def select_action(
    state: torch.tensor,
    steps_done: int,
    num_actions: int,
    policy_net: Union[DQN, DuelingDQN],
    device: torch.device,
    noisy_net: bool,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    min_start_steps: int,
    num_atoms: int,
    v_min: float,
    v_max: float,
):
    eps_threshold = schedule_epsilon(steps_done, noisy_net, eps_start, eps_end, eps_decay)
    if random.random() > eps_threshold and steps_done > min_start_steps:
        with torch.no_grad():
            if num_atoms == 1:
                return policy_net(state.to(device)).max(1)[1].view(1, 1)

            q_values_dist = policy_net(state.to(device)) * torch.linspace(v_min, v_max, num_atoms).to(device)
            return q_values_dist.sum(2).max(1)[1].view(1, 1)

    return torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)


def compute_cumulated_return(rewards: Union[List, deque], gamma: float) -> float:
    return sum([np.power(gamma, i) * transition.reward for i, transition in enumerate(rewards)])


def schedule_clip_epsilon(base_epsilon: float, steps_done: int, total_steps: int):
    return np.maximum(1.0 - (steps_done / total_steps), 0.1) * base_epsilon


def explained_variance(values: torch.Tensor, value_targets: torch.Tensor) -> float:
    return 1 - (value_targets - values).var() / value_targets.var()


def clip_gradients(net: torch.nn.Module, clip_const: float):
    for param in net.parameters():
        param.grad.data.clamp_(-clip_const, clip_const)


def polyak_averaging(target_net: torch.nn.Module, net: torch.nn.Module, tau: float):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def compute_grad_norm(net: torch.nn.Module) -> float:
    total_grad_norm = 0
    for params in net.parameters():
        if params.grad is not None:
            total_grad_norm += params.grad.data.norm(2).item()

    return total_grad_norm


def log_network_params(net: torch.nn.Module, writer: SummaryWriter, step: int, network_name: str):
    for tag, params in net.named_parameters():
        if params.grad is not None:
            writer.add_histogram(f"{network_name}/{tag}", params.data.cpu().numpy(), step)


def normalize_values(values: torch.Tensor, shift_mean: bool = False) -> torch.Tensor:
    if shift_mean:
        return (values - values.mean()) / values.std().clamp(min=1e-8)

    return values / values.std().clamp(min=1e-8)


class ValueStats:
    def __init__(self):
        self.counter = 0
        self.sum_values = 0
        self.sum_squared_values = 0
        self.running_mean = 0
        self.running_std = 0

    def update(self, values: torch.Tensor):
        self.counter += values.size(0)
        self.sum_values += values.sum().item()
        self.sum_squared_values += values.square().sum().item()

        self.running_mean = self.sum_values / self.counter
        self.running_std = np.sqrt((self.sum_squared_values / self.counter) - np.square(self.running_mean))

    def normalize(self, values: torch.Tensor, shift_mean: bool = False):
        self.update(values)

        if shift_mean:
            return (values - self.running_mean) / np.maximum(self.running_std, 1e-8)

        return values / np.maximum(self.running_std, 1e-8)
