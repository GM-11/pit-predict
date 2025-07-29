import torch


class PitstopModel(torch.nn.Module):
    """
    Neural network model for F1 pitstop prediction.

    This model predicts whether a driver should pit on the next lap
    based on current race conditions including tyre age, lap number,
    position, track status, and other factors.
    """

    def __init__(self, input_size):
        super(PitstopModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)
