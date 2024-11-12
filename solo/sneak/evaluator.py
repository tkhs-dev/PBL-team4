# 盤面を評価するニューラルネットワークを定義する
import os
import sys

import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.parameter_util import get_front_body, get_left_body, get_right_body, get_leftd_body, get_rightd_body, \
    get_snake_length, get_snake_health, get_snake_distance, get_snake_foods


class Evaluator:
    def __init__(self):
        self.model = EvaluatorModel()
        return

    @staticmethod
    def load(path):
        evaluator = Evaluator()
        evaluator.model.load(path)
        return evaluator

    def evaluate(self, game_state) -> float:
        return self.model(self.get_input_tensor(game_state)).item()

    @staticmethod
    def get_input_tensor(game_state : dict):
        return torch.Tensor(
            [
                get_front_body(game_state)/3,
                get_left_body(game_state)/3,
                get_right_body(game_state)/3,
                get_leftd_body(game_state)/3,
                get_rightd_body(game_state)/3,
                get_snake_length(game_state)/(game_state["board"]["width"]*game_state["board"]["height"]),
                get_snake_health(game_state)/100,
                get_snake_distance(game_state)/(game_state["board"]["height"]),
                get_snake_foods(game_state)/(game_state["board"]["height"]+game_state["board"]["width"])
            ]
        )

class EvaluatorModel(nn.Module):
    def __init__(self):
        super(EvaluatorModel, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        return

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path, weights_only=True))
        except:
            self.apply(self._initialize_with_random_weights)
            print("Failed to load model. Randomly initialized")
        return

    def save(self, path):
        torch.save(self.state_dict(), path)
        return

    def _initialize_with_random_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)  # 重みをHe初期化
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # バイアスをゼロで初期化