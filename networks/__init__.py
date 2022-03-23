from networks.q_networks import DQN, DuelingDQN


def get_network(
    network_name: str,
    width: int,
    height: int,
    num_actions: int,
    noisy_net: bool = False,
    noisy_std_init: float = 0.5,
    num_atoms: int = 1,
    num_fc_hidden_units: int = 256,
    num_channels: int = 64,
):
    if network_name == "DQN":
        return DQN(width, height, num_actions, noisy_net, noisy_std_init, num_atoms, num_fc_hidden_units, num_channels)
    elif network_name == "Dueling DQN":
        return DuelingDQN(
            width, height, num_actions, noisy_net, noisy_std_init, num_atoms, num_fc_hidden_units, num_channels
        )
    else:
        raise NotImplementedError(f"Network '{network_name}' not implemented!")
