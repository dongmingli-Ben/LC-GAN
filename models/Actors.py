import torch.nn as nn

class RealActor(nn.Module):

    def __init__(self):
        super(RealActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        output = self.layers(z)
        gates = self.sigmoid(output[:, :1024])
        dz = output[:, 1024:]
        result = (1-gates)*z + gates*dz
        return result