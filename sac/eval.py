import argparse
import os

import gym
from torch.utils.tensorboard import SummaryWriter

import sac_constants as const
from sac.sac_wrapper import SACWrapper
from utils import utils
import torch


def make_env(env_id: int):
    env = gym.make(const.ENV_NAME)
    env.seed(env_id)
    env.reset()
    return env


def main(policy_model_path: str):
    if not os.path.exists(policy_model_path):
        raise ValueError(f"Policy model file '{policy_model_path}' does not exist!")

    print(f"EVALUATE SAC MODEL {policy_model_path} FOR {const.EVAL_EPISODE_COUNT} EPISODES")

    utils.set_all_seeds(100)
    writer = SummaryWriter(log_dir=const.LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_env = make_env(100)
    init_screen = utils.get_pendulum_screen(eval_env, const.INPUT_SIZE)
    _, screen_height, screen_width = init_screen.shape
    action_dim = eval_env.action_space.shape[0]
    action_limits = torch.from_numpy(eval_env.action_space.high).to(device=device)

    sac_wrapper = SACWrapper(screen_width, screen_height, action_dim, action_limits, writer, policy_model_path)
    sac_wrapper.eval_policy(0, eval_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model_path", type=str, default="logs/best_policy_net.pth")
    parser_args = parser.parse_args()

    main(parser_args.policy_model_path)
