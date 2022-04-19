from typing import Tuple

import torch
from torch import nn
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.nn import functional as F

from networks.layers import CNNEncoder


class PolicyNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        action_dim: int,
        action_limits: torch.Tensor,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
        num_latent_dims: int = 64,
    ):
        super(PolicyNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_limits = action_limits

        self.encoder = CNNEncoder(width, height, num_frames, num_channels, num_latent_dims)
        self.fc_mean = nn.Linear(num_latent_dims, num_fc_hidden_units)
        self.layer_norm = nn.LayerNorm(num_fc_hidden_units)
        self.mean_head = nn.Linear(num_fc_hidden_units, action_dim)

        self.fc_log_std = nn.Linear(num_latent_dims, num_fc_hidden_units)
        self.layer_norm2 = nn.LayerNorm(num_fc_hidden_units)
        self.log_std_head = nn.Linear(num_fc_hidden_units, action_dim)

    def forward(self, x: torch.Tensor, detach_encoder: bool = False) -> Independent:
        x = self.encoder(x, detach=detach_encoder)

        x_mean = F.silu(self.layer_norm(self.fc_mean(x)))
        mean = self.mean_head(x_mean)

        x_log_std = F.silu(self.layer_norm2(self.fc_log_std(x)))
        log_std = self.log_std_head(x_log_std)
        std = torch.exp(torch.clamp(log_std, -20, 2))

        return Independent(Normal(loc=mean, scale=std), reinterpreted_batch_ndims=1)

    def get_action(self, x: torch.Tensor, eval_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.forward(x)
        action = torch.tanh(policy.sample())
        if eval_mode:
            action = torch.tanh(policy.mean)
        u = self.action_limits * action
        return action.detach(), u.detach()

    def evaluate(
        self, x: torch.Tensor, reparameterize: bool = True, detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Independent]:
        policy = self.forward(x, detach_encoder=detach_encoder)
        u = policy.rsample() if reparameterize else policy.sample()
        action = torch.tanh(u)
        log_prob = policy.log_prob(u) - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=1)

        return action, log_prob, policy


class QNet(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        action_dim: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
        num_latent_dims: int = 64,
        use_encoder: bool = False,
    ):
        super(QNet, self).__init__()

        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder = CNNEncoder(width, height, num_frames, num_channels, num_latent_dims)

        self.fc_q_value = nn.Linear(num_latent_dims + action_dim, num_fc_hidden_units)
        self.layer_norm = nn.LayerNorm(num_fc_hidden_units)
        self.q_value_head = nn.Linear(num_fc_hidden_units, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor, detach_encoder: bool = False) -> torch.Tensor:
        if self.use_encoder:
            x = self.encoder(x, detach=detach_encoder)

        x = torch.cat([x, a], dim=1)
        x = F.silu(self.layer_norm(self.fc_q_value(x)))
        x = self.q_value_head(x)

        return x


class Critic(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_frames: int,
        action_dim: int,
        num_fc_hidden_units: int = 256,
        num_channels: int = 64,
        num_latent_dims: int = 64,
    ):
        super(Critic, self).__init__()

        self.encoder = CNNEncoder(width, height, num_frames, num_channels, num_latent_dims)
        self.q_net1 = QNet(
            width,
            height,
            num_frames,
            action_dim,
            num_fc_hidden_units=num_fc_hidden_units,
            num_channels=num_channels,
            num_latent_dims=num_latent_dims,
            use_encoder=False,
        )
        self.q_net2 = QNet(
            width,
            height,
            num_frames,
            action_dim,
            num_fc_hidden_units=num_fc_hidden_units,
            num_channels=num_channels,
            num_latent_dims=num_latent_dims,
            use_encoder=False,
        )

    def forward(
        self, x: torch.Tensor, a: torch.Tensor, detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x, detach=detach_encoder)
        q1 = self.q_net1(x, a)
        q2 = self.q_net2(x, a)

        return q1, q2


class Curl(nn.Module):
    def __init__(self, num_latent_dims: int, critic: Critic, critic_target: Critic):
        super(Curl, self).__init__()

        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder

        self.W = nn.Parameter(torch.rand(num_latent_dims, num_latent_dims), requires_grad=True)

    def encode(self, x: torch.Tensor, detach: bool = False, target: bool = False) -> torch.Tensor:
        if target:
            with torch.no_grad():
                z = self.encoder_target(x)
        else:
            z = self.encoder(x)

        if detach:
            z = z.detach()

        return z

    def compute_logits(self, z_anchor: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        Wz = torch.matmul(self.W, z_target.T)
        logits = torch.matmul(z_anchor, Wz)
        return logits - torch.max(logits, dim=1)[0][:, None]
