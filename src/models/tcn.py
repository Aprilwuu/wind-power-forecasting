from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal Conv1D: output[t] only depends on <= t."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # remove future time steps caused by padding
        if self.padding != 0:
            return out[:, :, :-self.padding]
        return out


class TCNResidualBlock(nn.Module):
    """
    One residual block of TCN:
    Conv1d (causal) -> BN -> ReLU -> Dropout ->
    Conv1d (causal) -> BN -> ReLU -> Dropout -> Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # match channels for residual connection if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(out + identity)


class TCN(nn.Module):
    """
    TCN model for deterministic wind power forecasting.

    Input:  [B, seq_len, num_features]
    Output: [B, output_dim]  (e.g. horizon = 1)
    """
    def __init__(
        self,
        input_dim: int,     # number of input features
        output_dim: int,    # number of outputs (e.g.,horizons)
        channels: List[int] = [32, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(channels)

        for i in range(num_levels):
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            dilation = 2 ** i

            layers.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn = nn.Sequential(*layers)

        # final linear head: maps last hidden state -> prediction horizon
        self.fc = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, features]
        """
        # convert to CNN format
        t = x.transpose(1, 2)   # [B, C, T]

        y = self.tcn(t)         # [B, C_out, T]

        # take the last time step for forecasting
        last = y[:, :, -1]      # [B, C_out]

        return self.fc(last)    # [B, output_dim]

