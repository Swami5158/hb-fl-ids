import torch
import torch.nn as nn


class ConvMLP(nn.Module):
    """
    3-block 1D-CNN + MLP classifier for N-BaIoT intrusion detection.

    Architecture
    ------------
    Block 1: Conv1d(1->32,  k=3) + GroupNorm + ReLU + MaxPool1d(2)
    Block 2: Conv1d(32->64, k=3) + GroupNorm + ReLU + MaxPool1d(2)
    Block 3: Conv1d(64->128,k=3) + GroupNorm + ReLU + AdaptiveAvgPool1d(4)
    MLP    : Flatten -> Linear(512->128) -> ReLU -> Dropout -> Linear(128->11)

    Shape trace (input_dim=70):
        (B,70) -> unsqueeze(1) -> (B,1,70)
        Block1 -> (B,32,35)   [MaxPool: 70//2=35]
        Block2 -> (B,64,17)   [MaxPool: 35//2=17]
        Block3 -> (B,128,4)   [AdaptiveAvgPool(4)]
        Flatten -> (B,512)    [128*4=512]
        Linear  -> (B,128)
        Linear  -> (B,11)
        Total params: d = 98,571

    Design decisions
    ----------------
    GroupNorm over BatchNorm:
        BatchNorm uses batch-level statistics — unreliable in FL where each
        client has a small Non-IID batch. GroupNorm is independent of batch
        size and data distribution. Required for stable FL training.

    Conv1d kernel=3:
        N-BaIoT features are grouped by statistic type (weight, mean,
        variance always adjacent per time window). Conv1d exploits this
        local structure. Plain MLP treats all features as independent.

    Kaiming initialization:
        Prevents vanishing/exploding gradients from round 1.
    """

    def __init__(
        self,
        input_dim: int = 70,
        num_classes: int = 9,
        conv_channels: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pool_out: int = 4,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes
        gn_groups        = min(4, conv_channels)

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, conv_channels * 2), conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(conv_channels * 2, conv_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, conv_channels * 4), conv_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_out),
        )

        self.cnn_out_dim = pool_out * (conv_channels * 4)  # 4 * 128 = 512

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.cnn(x)
        return self.mlp(x)

    def get_gradient_dim(self) -> int:
        """Total trainable parameters d = 98,571."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_shapes(self):
        """
        Return list of shapes of all trainable parameter tensors.
        Used by HBFLAggregator for layer-wise processing.
        Order matches model.parameters() iteration — consistent with
        all gradient encoding/decoding throughout the codebase.
        """
        return [p.shape for p in self.parameters() if p.requires_grad]


def build_model(input_dim: int = 70, num_classes: int = 9) -> ConvMLP:
    model = ConvMLP(input_dim=input_dim, num_classes=num_classes)
    d = model.get_gradient_dim()
    print(f"[Model] Conv1D-MLP built — "
          f"input_dim={input_dim}, classes={num_classes}, params={d:,}")
    return model
