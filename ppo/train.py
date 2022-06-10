import random
from collections import deque
from itertools import count

import gym
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import ppo_constants as const
from networks.ppo_networks import PolicyNet, ValueNet
from ppo.ppo_wrappers import PPOWrapper
from utils import utils


def make_env(env_id: int):
    env = gym.make(const.ENV_NAME)
    env.seed(env_id)
    env.reset()
    return env


def collect_rollouts(
    actor_id: int,
    policy_net_old: PolicyNet,
    value_net_old: ValueNet,
    horizon: int,
    data_queue: mp.Queue,
    trainer_event: mp.Event,
    eval_event: mp.Event,
    device: torch.device,
    num_actions: int,
    num_episodes: int,
):
    env = make_env(actor_id)
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
                state_tensor.to(device)
                policy = policy_net_old(state_tensor.to(device))
                value = value_net_old(state_tensor.to(device))
                action = policy.sample()

            _, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], device=device)

            next_observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
            next_state = state.copy()
            next_state.append(next_observation)
            next_state_tensor = torch.stack(tuple(next_state), dim=1)

            data_queue.put(
                {
                    "env_id": actor_id,
                    "transition": (state_tensor, action, policy, reward, done, value),
                    "return": episode_return,
                    "steps_done": steps_done,
                    "episodes_done": episode,
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


def main():
    writer = SummaryWriter(log_dir=const.LOG_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmp_env = make_env(100)

    init_screen = utils.get_cartpole_screen(tmp_env, const.INPUT_SIZE)
    _, screen_height, screen_width = init_screen.shape
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    ppo_wrapper = PPOWrapper(screen_width, screen_height, num_actions, writer)

    manager = mp.Manager()
    batch_memory_queue = manager.Queue()
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

    finished_episodes = 0
    total_steps_done = 0

    while finished_episodes < const.NUM_EPISODES:
        rollout_data = batch_memory_queue.get(block=True)
        state, action, policy, reward, done, value = rollout_data["transition"]
        ppo_wrapper.batch_memory.add(rollout_data["env_id"], state, action, policy, reward, done, value)

        if done:
            finished_episodes += 1
            ppo_wrapper.episode_terminated(rollout_data["return"], total_steps_done)

        del rollout_data, state, action, policy, reward, done, value
        total_steps_done += 1

        if len(ppo_wrapper.batch_memory) % (const.HORIZON * const.NUM_ENVS) == 0 and len(ppo_wrapper.batch_memory) > 0:
            print(f"OPTIMIZE MODEL at STEP {total_steps_done}")
            ppo_wrapper.optimize_model(total_steps_done)
            ppo_wrapper.update_old_networks()
            ppo_wrapper.batch_memory.clear()

            for trainer_event in trainer_events:
                trainer_event.set()

        if total_steps_done % const.EVAL_FREQUENCY == 0 and finished_episodes > 0:
            for eval_event in eval_events:
                eval_event.set()

            ppo_wrapper.eval_policy(total_steps_done)

            for eval_event in eval_events:
                eval_event.clear()

    for actor_process in actor_processes:
        actor_process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
