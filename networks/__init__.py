from networks.q_networks import DQN, DuelingDQN


def get_network(network_name: str, width: int, height: int, num_actions: int):
    if network_name == "DQN":
        return DQN(width, height, num_actions)
    elif network_name == "Dueling DQN":
        return DuelingDQN(width, height, num_actions)
    else:
        raise NotImplementedError(f"Network '{network_name}' not implemented!")
