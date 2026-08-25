"""PyTorch models for identification, classification, and track prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CycloneImageNet(nn.Module):
    """Multi-task CNN on IR-like 128x128 imagery.

    Heads
    -----
    presence : cyclone vs non-cyclone
    pattern  : Dvorak-style cloud pattern (5 classes)
    category : IMD intensity class (8 classes)
    wind     : maximum sustained wind (knots)
    """

    def __init__(self, n_patterns: int = 5, n_categories: int = 8) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.b1 = ConvBlock(24, 48)
        self.b2 = ConvBlock(48, 96)
        self.b3 = ConvBlock(96, 160)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        hidden = 160
        self.presence = nn.Linear(hidden, 2)
        self.pattern = nn.Linear(hidden, n_patterns)
        self.category = nn.Linear(hidden, n_categories)
        self.wind = nn.Linear(hidden, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x).flatten(1)
        return self.drop(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.features(x)
        return {
            "presence": self.presence(h),
            "pattern": self.pattern(h),
            "category": self.category(h),
            "wind": self.wind(h).squeeze(-1),
        }


class TrackLSTM(nn.Module):
    """Sequence-to-sequence LSTM for 6-hourly track and intensity.

    Input  : (B, T_in, F) last observations
    Output : (B, T_out, 3)  [dlat_deg, dlon_deg, dwind_kt] per 6 h step
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden: int = 96,
        layers: int = 2,
        out_steps: int = 12,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.encoder = nn.LSTM(
            in_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden,
            hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
        self.seed = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.encoder(x)
        h = h_n[-1]
        token = self.seed(h).unsqueeze(1)
        hidden_state = (h.unsqueeze(0), torch.zeros_like(h).unsqueeze(0))
        outputs = []
        for _ in range(self.out_steps):
            token, hidden_state = self.decoder(token, hidden_state)
            outputs.append(self.head(token.squeeze(1)))
        return torch.stack(outputs, dim=1)


def image_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    lp = F.cross_entropy(pred["presence"], batch["presence"])
    lpat = F.cross_entropy(pred["pattern"], batch["pattern"])
    lcat = F.cross_entropy(pred["category"], batch["category"])
    lwind = F.smooth_l1_loss(pred["wind"], batch["wind"])
    return lp + lpat + 0.8 * lcat + 0.02 * lwind
