import torch.nn as nn

class RealClassifier(nn.Module):

    def __init__(self, num_outputs=1):
        super(RealClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_outputs),
            # nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, return_logit=False):
        logit = self.layers(x)
        if return_logit:
            return logit
        return self.sigmoid(logit)
