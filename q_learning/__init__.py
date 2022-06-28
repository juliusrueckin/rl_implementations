from typing import Union

from torch.utils.tensorboard import SummaryWriter

from q_learning.q_learning_wrappers import DeepQLearningWrapper, DoubleDeepQLearningWrapper


def get_q_learning_wrapper(
    double_q_learning: bool,
    screen_width: int,
    screen_height: int,
    num_actions: int,
    network_name: str,
    writer: SummaryWriter = None,
    policy_net_checkpoint: str = None,
    resume_training_checkpoint: str = None,
) -> Union[DeepQLearningWrapper, DoubleDeepQLearningWrapper]:
    if double_q_learning:
        return DoubleDeepQLearningWrapper(
            screen_width,
            screen_height,
            num_actions,
            network_name,
            writer,
            policy_net_checkpoint,
            resume_training_checkpoint,
        )
    else:
        return DeepQLearningWrapper(
            screen_width,
            screen_height,
            num_actions,
            network_name,
            writer,
            policy_net_checkpoint,
            resume_training_checkpoint,
        )
