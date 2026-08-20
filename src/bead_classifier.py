"""Two-head CNN for bead crops: digit class (0-8, in scope per current
labeling progress -- see src/check_label_balance.py) and rotation angle.

Rotation is predicted as a unit (sin, cos) vector rather than the
36-bins-of-10-degrees classification alternative from the original plan --
sin/cos regression handles the 359 -> 0 degree wraparound naturally, where
36-way classification would treat adjacent-in-reality bins 0 and 35 as
unrelated classes unless label-smoothed across the boundary.
"""

import torch
import torch.nn as nn

INPUT_SIZE = 64
NUM_DIGIT_CLASSES = 9  # digits 0-8; 9 excluded for now, too few labeled examples


class TwoHeadBeadNet(nn.Module):
    def __init__(self, num_digit_classes: int = NUM_DIGIT_CLASSES, in_channels: int = 3):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),  # 64 -> 32
            block(32, 64),           # 32 -> 16
            block(64, 128),          # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.digit_head = nn.Linear(128, num_digit_classes)
        self.rotation_head = nn.Linear(128, 2)  # (sin, cos)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.pool(self.features(x)).flatten(1)
        digit_logits = self.digit_head(feat)
        rot = self.rotation_head(feat)
        rot = rot / rot.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return digit_logits, rot
