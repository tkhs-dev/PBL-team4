# 盤面を評価するニューラルネットワークを定義する
from torch import nn

class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        return

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits