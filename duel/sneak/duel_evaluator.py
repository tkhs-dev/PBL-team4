# 盤面を評価するニューラルネットワークを定義する
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.parameter_util import get_snake_length, get_snake_health


class Evaluator:
    def __init__(self):
        self.model = EvaluatorModel()
        return

    @staticmethod
    def load(path):
        evaluator = Evaluator()
        evaluator.model.load(path)
        return evaluator

    def evaluate(self, game_state):
        return self.model(*map(lambda x:x.unsqueeze(0),self.get_input_tensor(game_state))).squeeze().tolist()

    @staticmethod
    def get_input_tensor(game_state : dict) -> (torch.Tensor, torch.Tensor):
        sneak0_body_board = np.zeros((11,11), dtype=np.int32)
        sneak0_head_board = np.zeros((11,11), dtype=np.int32)
        sneak0_tail_board = np.zeros((11,11), dtype=np.int32)
        sneak1_body_board = np.zeros((11,11), dtype=np.int32)
        sneak1_head_board = np.zeros((11,11), dtype=np.int32)
        sneak1_tail_board = np.zeros((11,11), dtype=np.int32)
        food_board = np.zeros((11,11), dtype=np.int32)
        you = game_state["you"]
        for body in you["body"]:
            if 0 <= body["x"] < 11 and 0 <= body["y"] < 11:
                sneak0_body_board[body["y"]][body["x"]] = 1
        if 0 <= you["head"]["x"] < 11 and 0 <= you["head"]["y"] < 11:
            sneak0_head_board[you["head"]["y"]][you["head"]["x"]] = 1
        sneak0_tail = you["body"][-1]
        if 0 <= sneak0_tail["x"] < 11 and 0 <= sneak0_tail["y"] < 11:
            sneak0_tail_board[sneak0_tail["y"]][sneak0_tail["x"]] = 1

        sneak1_length = 1
        if len(game_state["board"]["snakes"]) == 2:
            sneak1 = list(filter(lambda x:x['id']!=you['id'],game_state["board"]["snakes"]))[0]
            sneak1_length = len(sneak1["body"])
            for body in sneak1["body"]:
                if 0 <= body["x"] < 11 and 0 <= body["y"] < 11:
                    sneak1_body_board[body["y"]][body["x"]] = 1
            if 0 <= sneak1["head"]["x"] < 11 and 0 <= sneak1["head"]["y"] < 11:
                sneak1_head_board[sneak1["head"]["y"]][sneak1["head"]["x"]] = 1
            sneak1_tail = sneak1["body"][-1]
            if 0 <= sneak1_tail["x"] < 11 and 0 <= sneak1_tail["y"] < 11:
                sneak1_tail_board[sneak1_tail["y"]][sneak1_tail["x"]] = 1

        
        for food in game_state["board"]["food"]:
            food_board[food["y"]][food["x"]] = 1
        one_hot_encoded = np.stack([sneak0_body_board, sneak0_head_board, sneak0_tail_board, sneak1_body_board, sneak1_head_board, sneak1_tail_board, food_board], axis=0)
        board_tensor = torch.tensor(one_hot_encoded, dtype=torch.float32)
        game_tensor = torch.tensor([get_snake_health(game_state)/100,get_snake_length(game_state)/121,sneak1_length-get_snake_length(game_state)], dtype=torch.float32)
        return board_tensor,game_tensor


class EvaluatorModel(nn.Module):
    def __init__(self):
        super(EvaluatorModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 11 * 11 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        return

    def forward(self, board_input, sneak_input):
        cnn = self.cnn(board_input)
        cnn = cnn.view(-1, 64 * 11 * 11)
        input = torch.cat([cnn, sneak_input], dim=1)
        q = self.fc(input)
        return q

    def load(self, path):
        try:
            file = torch.load(path, weights_only=True)
            if file is dict:
                self.load_state_dict(file['model'])
            else:
                self.load_state_dict(file)
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