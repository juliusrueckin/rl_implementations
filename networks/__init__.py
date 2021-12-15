from networks.q_networks import DQN, DuelingDQN


def get_network(
    network_name: str, width: int, height: int, num_actions: int, noisy_net: bool = False, noisy_std_init: float = 0.5
):
    if network_name == "DQN":
        return DQN(width, height, num_actions, noisy_net, noisy_std_init)
    elif network_name == "Dueling DQN":
        return DuelingDQN(width, height, num_actions, noisy_net, noisy_std_init)
    else:
        raise NotImplementedError(f"Network '{network_name}' not implemented!")
