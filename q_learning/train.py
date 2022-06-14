import copy
import random
from collections import deque
from itertools import count
from typing import Union

import gym
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import q_learning_constants as const
from networks.q_networks import DuelingDQN, DQN
from q_learning import get_q_learning_wrapper
from utils import utils


def make_env(env_id: int):
    env = gym.make(const.ENV_NAME)
    env.seed(env_id)
    env.reset()
    return env


def collect_rollouts(
    actor_id: int,
    shared_policy_net: Union[DQN, DuelingDQN],
    data_queue: mp.Queue,
    learner_event: mp.Event,
    device: torch.device,
    num_episodes: int,
    num_actions: int,
):
    local_policy_net = copy.deepcopy(shared_policy_net).to(device)
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
        for t in count():
            while learner_event.is_set():
                continue

            local_policy_net.load_state_dict(shared_policy_net.state_dict())

            if t < no_op_steps:
                action = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
                _, _, done, _ = env.step(action.item())
                if done:
                    break

                continue

            if t % const.ACTION_REPETITIONS != 0:
                _, reward, done, _ = env.step(action.item())
                episode_return += reward

                if done:
                    data_queue.put(
                        {
                            "env_id": actor_id,
                            "transition": None,
                            "return": episode_return,
                            "steps_done": steps_done,
                            "episodes_done": episode,
                        }
                    )
                    break

                continue

            action = utils.select_action(
                state_tensor,
                steps_done,
                num_actions,
                local_policy_net,
                device,
                const.NOISY_NETS,
                const.EPS_START,
                const.EPS_END,
                const.EPS_DECAY,
                const.MIN_START_STEPS,
                const.NUM_ATOMS,
                const.V_MIN,
                const.V_MAX,
                eval_mode=False,
            )
            _, reward, done, _ = env.step(action.item())
            episode_return += reward
            reward = torch.tensor([reward], device=device)
            done_tensor = torch.tensor([int(done)], dtype=torch.int32, device=device)
            steps_done += 1

            next_observation = utils.get_cartpole_screen(env, const.INPUT_SIZE)
            next_state = state.copy()
            next_state.append(next_observation)
            next_state_tensor = torch.stack(tuple(next_state), dim=1)

            data_queue.put(
                {
                    "env_id": actor_id,
                    "transition": (state_tensor, action, next_state_tensor, reward, done_tensor),
                    "return": episode_return,
                    "steps_done": steps_done,
                    "episodes_done": episode,
                }
            )

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

    deep_q_learning_wrapper = get_q_learning_wrapper(
        const.DOUBLE_Q_LEARNING, screen_width, screen_height, num_actions, const.NETWORK_NAME, writer
    )

    replay_buffer_queue = mp.Queue()
    learner_events = [mp.Event() for _ in range(const.NUM_ENVS)]
    actor_processes = []

    for env_id in range(const.NUM_ENVS):
        actor_process = mp.Process(
            target=collect_rollouts,
            args=(
                env_id,
                deep_q_learning_wrapper.policy_net,
                replay_buffer_queue,
                learner_events[env_id],
                device,
                int(const.NUM_EPISODES / const.NUM_ENVS) + 1,
                num_actions,
            ),
            daemon=True,
        )
        actor_process.start()
        actor_processes.append(actor_process)

    finished_episodes = 0
    total_steps_done = 0

    while finished_episodes < const.NUM_EPISODES:
        rollout_data = replay_buffer_queue.get(block=True)
        if rollout_data["transition"] is not None:
            state, action, next_state, reward, done = rollout_data["transition"]
            deep_q_learning_wrapper.replay_buffer.push(
                rollout_data["env_id"],
                copy.deepcopy(state),
                copy.deepcopy(action),
                copy.deepcopy(next_state),
                copy.deepcopy(reward),
                copy.deepcopy(done),
            )

            if done.item():
                finished_episodes += 1
                deep_q_learning_wrapper.episode_terminated(rollout_data["return"], total_steps_done)

            del rollout_data, state, action, next_state, reward, done
        else:
            finished_episodes += 1
            deep_q_learning_wrapper.episode_terminated(rollout_data["return"], total_steps_done)

            del rollout_data

        total_steps_done += 1

        if total_steps_done % const.EVAL_FREQUENCY == 0 and finished_episodes > 0:
            for learner_event in learner_events:
                learner_event.set()

            deep_q_learning_wrapper.eval_policy(total_steps_done)

            for learner_event in learner_events:
                learner_event.clear()

        if total_steps_done >= const.MIN_START_STEPS and total_steps_done % const.OPTIMIZATION_UPDATE == 0:
            for learner_event in learner_events:
                learner_event.set()

            deep_q_learning_wrapper.optimize_model(total_steps_done)

            for learner_event in learner_events:
                learner_event.clear()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
