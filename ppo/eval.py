import argparse
import os

from torch.utils.tensorboard import SummaryWriter

import q_learning_constants as const
from ppo.ppo_wrappers import PPOWrapper
from utils import utils


def main(policy_model_path: str):
    if not os.path.exists(policy_model_path):
        raise ValueError(f"Policy model file '{policy_model_path}' does not exist!")

    print(f"EVALUATE PPO MODEL {policy_model_path} FOR {const.EVAL_EPISODE_COUNT} EPISODES")

    utils.set_all_seeds(100)
    writer = SummaryWriter(log_dir=const.LOG_DIR)

    eval_env = utils.make_env(100, const.ENV_NAME)
    init_screen = utils.get_cartpole_screen(eval_env, const.INPUT_SIZE)
    _, screen_height, screen_width = init_screen.shape
    num_actions = eval_env.action_space.n

    ppo_wrapper = PPOWrapper(screen_width, screen_height, num_actions, writer, policy_model_path)
    ppo_wrapper.eval_policy(eval_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model_path", type=str, default="logs/best_policy_net.pth")
    parser_args = parser.parse_args()

    main(parser_args.policy_model_path)
