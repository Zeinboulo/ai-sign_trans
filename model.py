from typing import Optional, Tuple

import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 225,
        use_projection: bool = True,
        projection_dim: int = 512,
    ) -> None:
        super().__init__()

        self.use_projection = use_projection
        self.projection_dim = projection_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )

        # Infer flattened dim from a dummy forward to handle pooling behavior.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = self.cnn(dummy)
            flattened_dim = out.shape[1] * out.shape[2]

        if use_projection:
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.cnn_out_dim = projection_dim
        else:
            self.projection = nn.Flatten()
            self.cnn_out_dim = flattened_dim

        self.bilstm1 = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.bilstm2 = nn.LSTM(
            input_size=512,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, f = x.shape
        x = x.view(b * t, 1, f)
        x = self.cnn(x)
        x = self.projection(x)
        x = x.view(b, t, -1)
        x, _ = self.bilstm1(x)
        x = self.dropout(x)
        x, _ = self.bilstm2(x)
        if lengths is not None:
            lengths = lengths.to(x.device)
            lengths = torch.clamp(lengths, min=1, max=t)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
            x = x.gather(1, idx).squeeze(1)
        else:
            x = x[:, -1, :]
        x = self.classifier(x)
        return x


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
