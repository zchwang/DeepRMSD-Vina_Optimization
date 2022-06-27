import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, rate):
        super(CNN, self).__init__()

        # self.flatten = torch.flatten()  # 128 * 7 * 210
        self.fc1 = nn.Sequential(
            nn.Linear(1470, 1024),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(1024),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(512),
            )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(256),
            )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(128),
        )

        self.fc5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=rate),
            nn.BatchNorm1d(64),
        )

        self.out = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        out = self.out(x)

        return out