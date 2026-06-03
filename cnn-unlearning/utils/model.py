import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(64*24*24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x)) # Bx1x28x28  => Bx32x26x26
        x = self.relu(self.conv2(x)) # Bx32x26x26 => Bx64x24x24

        x = self.flatten(x)

        x = self.relu(self.fc1(x))

        return self.fc2(x)
    