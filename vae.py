import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

# torch.log(0)によるnanを防ぐ
def torch_log(x):
    return torch.log(torch.clamp(x, min=1e-10))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAEモデルの実装
class VAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()

        # Encoder with CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: [batch, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),  # Flatten for the dense layer
        )

        self.dense_encmean = nn.Linear(32 * 7 * 7, z_dim)
        self.dense_encvar = nn.Linear(32 * 7 * 7, z_dim)

        # Decoder with Transpose CNN
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 1, 28, 28]
            nn.Sigmoid()
        )

    def _encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x) # 1. (1,28,28) -> 32*7*7
        mean = self.dense_encmean(x) # 5. 32*7*7 -> z_dim(linear)[mean]
        std = F.softplus(self.dense_encvar(x)) # 5. 32*7*7 -> z_dim(linear)[std], softplusはreluの連続関数版

        return mean, std # z_dim, z_dim

    def _sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.training:
            # 再パラメータ化トリック．この乱数は計算グラフで勾配の通り道に無い．
            epsilon = torch.randn(mean.shape).to(device) # epsilon: shape(batch, z_dim)
            return mean + std * epsilon # shape(batch, z_dim)
        else:
            return mean

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        # input shape: (batch, z_dim)
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x) # (1,28,28) -> z_dim, z_dim
        z = self._sample_z(mean, std) # z_dim, zdim -> z_dim
        x = self._decoder(z) # z_dim -> (1,28,28)
        return x, z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._encoder(x) # 1. (1,28,28) -> z_dim, zdim

        # KL loss(正則化項)の計算. mean, stdは (batch_size , z_dim)
        # 0.5 * Σ(1 + log(std^2) - mean^2 - std^2)
        # torch.meanはbatch_sizeに対してtorch.sumはz_dimに対して
        KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std**2) - mean**2 - std**2, dim=1))

        z = self._sample_z(mean, std) # z_dim, zdim -> z_dim
        y = self._decoder(z) # z_dim -> (1,28,28)

        # reconstruction loss(負の再構成誤差)の計算. x, yともに (batch_size , 784)
        # torch.sumは上式のD(=784)に関するもの. torch.meanはbatch_sizeに関するもの.
        # reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=1))
        reconstruction = torch.mean(torch.sum(x * torch_log(y) + (1 - x) * torch_log(1 - y), dim=[1, 2, 3]))

        return KL, -reconstruction