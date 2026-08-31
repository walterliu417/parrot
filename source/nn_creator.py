import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, 8, 8)

        b, c, _, _ = x.shape

        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)

        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, channels=128, use_se=True):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels) if use_se else nn.Identity()

        self.act2 = nn.SiLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out = out + residual
        out = self.act2(out)

        return out


class ChessEvaluationNet(nn.Module):
    def __init__(
        self,
        channels=128,
        num_blocks=8,
        value_channels=32,
        hidden_size=256,
        use_se=True,
    ):
        super().__init__()

        # Input:
        # (B, 2, 8, 8)
        #
        # Output:
        # (B, 1), in [0, 1]

        # --------------------------------------------------
        # Stem
        # --------------------------------------------------

        self.stem = nn.Sequential(
            nn.Conv2d(
                2,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        # --------------------------------------------------
        # Residual tower
        # --------------------------------------------------

        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(
                    channels=channels,
                    use_se=use_se,
                )
                for _ in range(num_blocks)
            ]
        )

        # --------------------------------------------------
        # Value head
        # --------------------------------------------------

        self.value_head = nn.Sequential(
            nn.Conv2d(
                channels,
                value_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(value_channels),
            nn.SiLU(),

            nn.Flatten(),

            nn.Linear(
                value_channels * 8 * 8,
                hidden_size,
            ),
            nn.SiLU(),

            nn.Linear(hidden_size, 1),

            # Evaluation between 0 and 1
            nn.Sigmoid(),
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.stem(x)
        x = self.residual_tower(x)
        x = self.value_head(x)

        return x