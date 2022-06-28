import argparse
import copy
import random
from collections import deque
from itertools import count

import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import ppo_constants as const
from networks.ppo_networks import PolicyNet, ValueNet
from ppo.ppo_wrappers import PPOWrapper
from utils import utils


def collect_rollouts(
    actor_id: int,
    shared_policy_net_old: PolicyNet,
    shared_value_net_old: ValueNet,
    horizon: int,
    data_queue: mp.Queue,
    trainer_event: mp.Event,
    eval_event: mp.Event,
    device: torch.device,
    num_actions: int,
    num_episodes: int,
):
    utils.set_all_seeds(actor_id)
    local_policy_net_old = copy.deepcopy(shared_policy_net_old).to(device)
    local_value_net_old = copy.deepcopy(shared_value_net_old).to(device)
    env = utils.make_env(actor_id, const.ENV_NAME)
    steps_done = 0

    for episode in range(num_episodes):
        env.reset()
        observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
        state = deque(
            [torch.zeros(observation.size()) for _ in range(const.FRAMES_STACKED)], maxlen=const.FRAMES_STACKED
        )
        state.append(observation)
        state_tensor = torch.stack(tuple(state), dim=1)

        episode_return = 0
        no_op_steps = random.randint(0, const.NO_OP_MAX_STEPS)
        for step in count():
            while eval_event.is_set():
                continue

            steps_done += 1
            if step < no_op_steps:
                action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
                _, _, done, _ = env.step(action.item())
                if done:
                    break

                continue

            if step % const.ACTION_REPETITIONS != 0:
                _, _, done, _ = env.step(action.item())
                if done:
                    break

                continue

            with torch.no_grad():
                policy = local_policy_net_old(state_tensor.to(device))
                action = policy.sample()

            _, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], device=device)

            next_observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
            next_state = state.copy()
            next_state.append(next_observation)
            next_state_tensor = torch.stack(tuple(next_state), dim=1)

            last_value = torch.zeros(1, device=device)
            if steps_done % horizon == 0 and not done:
                with torch.no_grad():
                    last_value = local_value_net_old(next_state_tensor.to(device))

            data_queue.put(
                {
                    "env_id": actor_id,
                    "transition": (state_tensor, action, policy, reward, done),
                    "return": episode_return,
                    "steps_done": steps_done,
                    "episodes_done": episode,
                    "last_value": last_value,
                }
            )

            if steps_done % horizon == 0:
                trainer_event.wait()
                trainer_event.clear()

            if done:
                break

            state_tensor = next_state_tensor
            state = next_state.copy()

    env.close()


def main(resume_training_checkpoint: str = None):
    utils.set_all_seeds(100)
    writer = SummaryWriter(log_dir=const.LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_env = utils.make_env(100, const.ENV_NAME)

    init_screen = utils.get_cartpole_screen(tmp_env, const.INPUT_SIZE)
    _, screen_height, screen_width = init_screen.shape
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    ppo_wrapper = PPOWrapper(screen_width, screen_height, num_actions, writer, None, resume_training_checkpoint)

    batch_memory_queue = mp.Queue()
    trainer_events = [mp.Event() for _ in range(const.NUM_ENVS)]
    eval_events = [mp.Event() for _ in range(const.NUM_ENVS)]
    actor_processes = []

    for env_id in range(const.NUM_ENVS):
        actor_process = mp.Process(
            target=collect_rollouts,
            args=(
                env_id,
                ppo_wrapper.policy_net_old,
                ppo_wrapper.value_net_old,
                const.HORIZON,
                batch_memory_queue,
                trainer_events[env_id],
                eval_events[env_id],
                device,
                num_actions,
                int(const.NUM_EPISODES / const.NUM_ENVS) + 1,
            ),
            daemon=True,
        )
        actor_process.start()
        actor_processes.append(actor_process)

    while ppo_wrapper.finished_episodes < const.NUM_EPISODES:
        rollout_data = batch_memory_queue.get(block=True)
        state, action, policy, reward, done = rollout_data["transition"]
        env_id = rollout_data["env_id"]
        last_value = rollout_data["last_value"]
        ppo_wrapper.batch_memory.add(
            copy.deepcopy(env_id),
            copy.deepcopy(state),
            copy.deepcopy(action),
            copy.deepcopy(policy),
            copy.deepcopy(reward),
            copy.deepcopy(done),
            copy.deepcopy(last_value),
        )

        if done:
            ppo_wrapper.episode_terminated(rollout_data["return"])

        del rollout_data, state, action, policy, reward, done, env_id, last_value
        ppo_wrapper.total_steps_done += 1

        if len(ppo_wrapper.batch_memory) % (const.HORIZON * const.NUM_ENVS) == 0 and len(ppo_wrapper.batch_memory) > 0:
            ppo_wrapper.optimize_model()

            for trainer_event in trainer_events:
                trainer_event.set()

        if ppo_wrapper.total_steps_done % const.EVAL_FREQUENCY == 0 and ppo_wrapper.finished_episodes > 0:
            for eval_event in eval_events:
                eval_event.set()

            ppo_wrapper.eval_policy(utils.make_env(100, const.ENV_NAME))

            for eval_event in eval_events:
                eval_event.clear()

    for actor_process in actor_processes:
        actor_process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_training", type=str, default=None)
    parser_args = parser.parse_args()

    main(parser_args.resume_training)
