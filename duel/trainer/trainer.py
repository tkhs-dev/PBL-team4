import json
import signal
import sys
import threading
import os
import io
import traceback
import random
from datetime import datetime
from time import sleep

from logging import Logger

import numpy as np
import torch
from early_stopping_pytorch import EarlyStopping
from torch import optim, nn
from torch.utils.data import Dataset

from api_client import TestApiClient, ApiClientImpl
from duel.sneak.duel_evaluator import Evaluator, EvaluatorModel
from duel.sneak.duel_player import AIPlayer
from duel.trainer.api_client import ApiClient
from duel.trainer.common import file_exists_and_not_empty, get_version_str
from game_downloader import GameDownloader
from shared import rule
from shared.embedded_rules import GameSettings, Client, start_duel_game
from shared.rule import Direction, TurnResult, move, is_move_maybe_safe

version = "1.0.0"
version_str = get_version_str(version)

class CancelToken:
    def __init__(self):
        self._cancel_requested = False

    def request_cancel(self):
        """キャンセルをリクエストする"""
        self._cancel_requested = True

    def is_cancellation_requested(self):
        """キャンセルがリクエストされたか確認する"""
        return self._cancel_requested

# ニューラルネットワークの学習用データセット
class DuelDataset(Dataset):
    def __init__(self, data):
        self.datas = []
        for d in data:
            state = d['game_state']
            turn = {
                'input': Evaluator.get_input_tensor(state),
            }
            action = d['action']
            target = list(map(
                lambda x:  -1 if rule.move(state, x)[0]==TurnResult.LOSE else 0,
                Direction
            ))
            if action == Direction.UP:
                target[0] += 1
            elif action == Direction.DOWN:
                target[1] += 1
            elif action == Direction.LEFT:
                target[2] += 1
            elif action == Direction.RIGHT:
                target[3] += 1
            turn['target'] = torch.tensor(target, dtype=torch.float32)
            for i in [0, 90, 180, 270]:
                rotated = {
                    'input': (rotate_board_tensor(turn['input'][0], i),turn['input'][1]),
                   'target': torch.roll(turn['target'], shifts={90: 2, 180: 3, 270: 1}.get(i % 360, 0), dims=0)
                }
                self.datas.append(rotated)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        input_tensor = self.datas[idx]['input']
        target = self.datas[idx]['target']
        return input_tensor, target

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def sample(self):
        s = random.uniform(0, self.total())
        idx, p, data = self.get(s)
        return idx, p, data

class ReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, rotation=False):
        # if rotation is True, the memory will store 4 times more data by rotating the board
        capacity = capacity * 4 if rotation else capacity
        self.memory = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.0001
        self.max_priority = 100
        self.rotation = rotation

    def push(self, board_state, game_state, action, reward, next_board_state, next_game_state, done):
        if self.rotation:
            for angle in [0, 90, 180, 270]:
                self.memory.add(self.max_priority,(rotate_board_tensor(board_state, angle), game_state, rotate_action(action, angle), reward, rotate_board_tensor(next_board_state, angle), next_game_state, done))
        else:
            self.memory.add(self.max_priority,(board_state, game_state, action, reward, next_board_state, next_game_state, done))


    def sample(self, batch_size, step_progress):
        total = self.memory.total()
        beta = self.beta
        for i in range(batch_size):
            idx, p, data = self.memory.sample()
            weight = (1/(total*p)) ** (beta + (1-beta) * step_progress)
            yield idx, weight, data

    def update(self, idx, error):
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.memory.update(idx, p)

    def __len__(self):
        return self.memory.write

rotation_map = {
    0: [0, 1, 2, 3],   # 0度: そのまま
    90: [3, 2, 0, 1],  # 90度: UP→RIGHT, DOWN→LEFT, LEFT→UP, RIGHT→DOWN
    180: [1, 0, 3, 2], # 180度: UP→DOWN, DOWN→UP, LEFT→RIGHT, RIGHT→LEFT
    270: [2, 3, 1, 0]  # 270度: UP→LEFT, DOWN→RIGHT, LEFT→DOWN, RIGHT→UP
}
def rotate_action(action:torch.Tensor, angle = 90):
    action = action.item()
    return torch.tensor(rotation_map[angle][action])

def rotate_actions_batch(actions:torch.Tensor, angle = 90):
    return torch.tensor([rotation_map[angle][action.item()] for action in actions])

def rotate_board_tensor(board_state:torch.Tensor, angle = 90):
    k = angle // 90
    return torch.rot90(board_state, k, [2,1])

def rotate_board_tensors_batch(board_states:torch.Tensor, angle = 90):
    k = angle // 90
    return torch.rot90(board_states, k, [3,2])


class Trainer:
    def __init__(self, logger, api_client:ApiClient, device, cancel_token:CancelToken):
        self.logger = logger
        self.api_client = api_client
        self.device = device
        self.cancel_token = cancel_token

    #モデルをロードする. キャッシュがあればそれを使う.
    def load_model(self, model_id: str):
        data = self.api_client.get_model_bytes(model_id)
        return torch.load(io.BytesIO(data), map_location=self.device, weights_only=True)

    def start(self, task):
        pass

class SupervisedTrainer(Trainer):
    def __init__(self, logger, api_client:ApiClient, device, cancel_token:CancelToken):
        super().__init__(logger, api_client, device, cancel_token)

    def task_supervised(self, base_model:EvaluatorModel, optimizer:torch.optim.Adam, task, batch_size = 64) -> bytes | None:
        print("Supervised training started")
        games = task['parameters']['games']
        downloader = GameDownloader()
        datas = []
        for game in games:
            if file_exists_and_not_empty(f"./cache/games/{game}.json"):
                with open(f"./cache/games/{game}.json", "r") as f:
                    data= json.load(f)
            else:
                player_id,game_id = game.split("_")
                data = downloader.download_data(game_id, player_id)
                with open(f"./cache/games/{game}.json", "w") as f:
                    f.write(json.dumps(data))
            #すべてのゲームについてデータをダウンロードし, 最後のターン以外をdatasに追加
            datas += data[:-1]

        dataset = DuelDataset(datas)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = base_model.to(self.device)

        if not optimizer:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to('cuda')

        criterion = torch.nn.MSELoss()
        epoch = int(task['parameters']['epochs'])

        #学習ループ
        early_stopping = EarlyStopping(patience=7, verbose=False, trace_func=self.logger.debug)
        for e in range(epoch):
            if self.cancel_token.is_cancellation_requested():
                return None
            model.train()
            total_loss = 0
            for (inputs, targets) in dataloader:
                board_input, game_input, targets = inputs[0].to(self.device), inputs[1].to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(board_input, game_input) #inputsを展開して渡す
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            early_stopping(total_loss/len(dataloader), model)
            if early_stopping.early_stop:
                self.logger.debug(f"Early stopping at epoch {e+1}/{epoch}")
                break
            self.logger.debug(f"Epoch {e+1}/{epoch} Loss: {total_loss/len(dataloader)}")
        #学習結果をバイナリに変換して返す
        model.eval()
        buffer = io.BytesIO()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, buffer)
        return buffer.getvalue()

    def load_model(self, model_id: str):
        data = super().load_model(model_id)
        model = EvaluatorModel()
        model.load_state_dict(data['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(data['optimizer'])
        return model,optimizer

    def start(self, task):
        if not os.path.exists("./cache/games"):
            os.makedirs("./cache/games")

        #ベースモデルが指定されている場合はロード, 指定されていない新規作成
        if task['baseModelId'] is None:
            model = EvaluatorModel()
            optimizer = None
        else:
            model,optimizer = self.load_model(task['baseModelId'])
            if model is None:
                raise Exception(f"Failed to load model({task['baseModelId']})")

        return self.task_supervised(model, optimizer, task)

# ハイパーパラメータ
GAMMA = 0.99  # 割引率
LR = 0.0001  # 学習率
BATCH_SIZE = 32
EPSILON_START = 1
EPSILON_END = 0.1
EPISODES = 100000
EPSILON_DECAY = 300000 # steps
MEMORY_CAPACITY = 1000000 # steps
TARGET_UPDATE = 200 # episodes
START_STEP = 10000 # steps
SAVE_STEP = 100 # episodes
ROTATION = True

class ReinforcementTrainer(Trainer):

    def __init__(self, logger, api_client:ApiClient, device, cancel_token:CancelToken):
        super().__init__(logger, api_client, device, cancel_token)
        self.device = device
        self.policy_net = EvaluatorModel().to(device)
        self.target_net = EvaluatorModel().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = None
        self.memory = ReplayMemory(MEMORY_CAPACITY,ROTATION)
        self.total_reward = 0
        self.steps_done = 0

        self.last_you = {}
        self.last_opponent = {}
        self.last_you_stack = False
        self.last_opponent_stack = False

    # ε-greedyポリシー
    def select_action(self, game_state, state, n_actions):
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * self.steps_done / EPSILON_DECAY)
        if (self.steps_done > START_STEP) and (random.random() > epsilon):
            with torch.no_grad():
                return self.policy_net(*map(lambda x:x.unsqueeze(0).to(self.device),state)).argmax(dim=1).item()  # 最適な行動
        else:
            choice = range(n_actions)
            return random.choice(list(choice))  # ランダムな行動

    # 学習ステップ
    def optimize_model(self):
        if self.steps_done < START_STEP:
            return
        if len(self.memory) < BATCH_SIZE:
            return
        samples = self.memory.sample(BATCH_SIZE, self.steps_done / MEMORY_CAPACITY)
        # タプルのリストを別々のリストに分解
        idxs = []
        transitions = []
        weights = []
        for idx, weight, data in samples:
            idxs.append(idx)
            transitions.append(data)
            weights.append(weight)

        batch = list(zip(*transitions))
        board_state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        game_state_batch = torch.tensor(np.array(batch[1]), dtype=torch.float32).to(self.device)

        action_batch = torch.tensor(batch[2]).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch[3], dtype=torch.float32).unsqueeze(1).to(self.device)
        reward_batch = reward_batch.clamp(-1, 1)
        next_board_state_batch = torch.tensor(np.array(batch[4]), dtype=torch.float32).to(self.device)
        next_game_state_batch = torch.tensor(np.array(batch[5]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[6], dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q値を計算
        state_action_values = self.policy_net(board_state_batch,game_state_batch).gather(1, action_batch)
        next_action = self.policy_net(next_board_state_batch,next_game_state_batch).max(1)[1].unsqueeze(1)
        next_state_values = self.target_net(next_board_state_batch,next_game_state_batch).gather(1, next_action)
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

        # 損失関数
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        weighted_loss = loss * torch.tensor(weights, dtype=torch.float32).to(self.device)
        for i in range(BATCH_SIZE):
            self.memory.update(idxs[i], weighted_loss[i].item())

        # ネットワーク更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_reward(self, game_state):
        if game_state["turn"] > 0 and len(game_state["board"]["snakes"]) == 2:
            you = game_state["you"]
            opponent = list(filter(lambda x: x['id'] != you["id"], game_state["board"]["snakes"]))[0]
            reward = 0
            if game_state["you"]["health"] == 100:
                # エサを食べた時
                reward = 1
            else:
                head_y = you["head"]
                head_o = opponent["head"]
                dx = head_y["x"] - head_o["x"]
                dy = head_y["y"] - head_o["y"]
                if np.abs(dx) == 1:
                    # 相手の頭とx座標が隣接している場合
                    if head_o["x"] == 0:
                        # 相手が壁に隣接している場合
                        if {"x": 1, "y": head_o["y"]} in you["body"][1:]:
                            # 相手の頭の右に自分の体がある場合
                            reward = 1
                        else:
                            # 相手の頭の右に自分の体がない場合
                            if np.abs(dy) < 3:
                                if you["length"] <= opponent["length"]:
                                    reward = -1
                                else:
                                    reward = 0.5
                    elif head_o["x"] == 10:
                        # 相手が壁に隣接している場合
                        if {"x": 9, "y": head_o["y"]} in you["body"][1:]:
                            # 相手の頭の左に自分の体がある場合
                            reward = 1
                        else:
                            # 相手の頭の左に自分の体がない場合
                            if np.abs(dy) < 3:
                                if you["length"] <= opponent["length"]:
                                    reward = -1
                                else:
                                    reward = 0.5
                    elif np.abs(dy) == 1:
                        if you["length"] <= opponent["length"]:
                            reward = -1
                        else:
                            reward = 1

                elif np.abs(dy) == 1:
                    # 相手の頭とy座標が隣接している場合
                    if head_o["y"] == 0:
                        # 相手が壁に隣接している場合
                        if {"x": head_o["x"], "y": 1} in you["body"][1:]:
                            # 相手の頭の下に自分の体がある場合
                            reward = 1
                        else:
                            # 相手の頭の下に自分の体がない場合
                            if np.abs(dx) < 3:
                                if you["length"] <= opponent["length"]:
                                    reward = -1
                                else:
                                    reward = 0.5
                    elif head_o["y"] == 10:
                        # 相手が壁に隣接している場合
                        if {"x": head_o["x"], "y": 9} in you["body"][1:]:
                            # 相手の頭の上に自分の体がある場合
                            reward = 1
                        else:
                            # 相手の頭の上に自分の体がない場合
                            if np.abs(dx) < 3:
                                if you["length"] <= opponent["length"]:
                                    reward = -1
                                else:
                                    reward = 0.5
                    elif np.abs(dx) == 1:
                        if you["length"] <= opponent["length"]:
                            reward = -1
                        else:
                            reward = 1
                elif opponent["length"] < you["length"]:
                    # 相手の体が自分の体より短い場合
                    if len(game_state["you"]["body"]) -len(opponent["body"]) < 8:
                        reward = np.exp(-1. * game_state["turn"]/100) + 1
                    else:
                        reward = -0.01
                else:
                    reward = -0.1

            return reward
        return 0

    def calculate_result_reward(self, result, turn, kill, cause, opponent_stack):
        reward = 0
        if result == "win":
            if turn < 40:
                if kill != "head-collision":
                    reward = 5
            else:
                if kill != "head-collision":
                    reward = 5
                elif opponent_stack:
                    reward = 7
                else:
                    reward = 2
        elif result == "lose":
            if cause == "wall-collision":
                reward = -10
            elif cause == "snake-self-collision":
                reward = -10
            elif cause == "snake-collision":
                reward = -10
            elif cause == "head-collision":
                reward = -10
            else:
                reward = 10
        else:
            reward = -10
        return reward


    def start_callback(self, game_state):
        self.total_reward = 0
        return 0

    def move_callback(self, game_state):
        next_state_tensor = Evaluator.get_input_tensor(game_state)
        if game_state["turn"] > 0 and len(game_state["board"]["snakes"]) == 2:
            last_state_tensor = self.last_you["state"]
            last_action = self.last_you["action"]

            reward = self.calculate_reward(game_state)
            self.total_reward += reward
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            self.memory.push(last_state_tensor[0], last_state_tensor[1], last_action, reward, next_state_tensor[0], next_state_tensor[1], False)
            self.optimize_model()

        if len(game_state["board"]["snakes"]) != 2:
            return Direction.UP

        self.last_you["state"] = next_state_tensor
        action = self.select_action(game_state, next_state_tensor, 4)
        if not is_move_maybe_safe(game_state, Direction.by_index(action)):
            # 危険な行動をとろうとした場合、そのことを記憶、学習して安全な行動に置き換える
            self.memory.push(next_state_tensor[0], next_state_tensor[1], action, torch.tensor([-10], dtype=torch.float32).to(self.device), next_state_tensor[0], next_state_tensor[1], True)
            self.optimize_model()
            safe_moves = list(filter(lambda x: is_move_maybe_safe(game_state, x), Direction))
            if len(safe_moves) > 0:
                action = Direction.index(random.choice(safe_moves))
                self.last_you_stack = False
            else:
                self.last_you_stack = True

        self.steps_done += 1
        self.last_you["action"] = action
        return Direction.by_index(action)

    def end_callback(self, game_state):
        return 0

    def opponent_move_callback(self, game_state, move_cb):
        state_tensor = Evaluator.get_input_tensor(game_state)
        move = move_cb(game_state)
        if game_state["turn"] > 0:
            last_state_tensor = self.last_opponent["state"]
            last_action = self.last_opponent["action"]
            reward = self.calculate_reward(game_state)
            self.memory.push(last_state_tensor[0], last_state_tensor[1], last_action, torch.tensor([reward], dtype=torch.float32).to(self.device), state_tensor[0], state_tensor[1], False)

        self.last_opponent_stack = not is_move_maybe_safe(game_state, move)
        if len(game_state["board"]["snakes"]) == 2:
            self.last_opponent = {
                "state": state_tensor,
                "action": Direction.index(move)
            }
        return move


    def finalize(self):
        print("Training finished!")
        # モデルの保存
        evaluator = Evaluator()
        evaluator.model = self.policy_net
        evaluator.model.save("./checkpoint.pth")
        print("Saved model")

    def task_reinforced(self, initial_episode):
        # トレーニングループ
        num_episodes = EPISODES
        opponent = Evaluator()
        episode = initial_episode
        turns = []
        for eps in range(num_episodes - initial_episode):
            episode += 1
            evaluator = Evaluator()
            evaluator.model = self.policy_net
            evaluator.model.eval()
            client1 = Client()
            client1.on_start = self.start_callback
            client1.on_move = lambda game_state: self.move_callback(game_state)
            client1.on_end = self.end_callback
            if random.random() <= 5:
                opponent.model.load_state_dict(self.target_net.state_dict())
            else:
                opponent.model.load("C:\\Users\\tokuh\\OneDrive - OUMail (Osaka University)\\checkpoint4.pth")
            player = AIPlayer(opponent, True)
            client2 = Client()
            client2.on_start = player.on_start
            client2.on_move = lambda game_state: self.opponent_move_callback(game_state, player.on_move)
            client2.on_end = player.on_end
            setting = GameSettings()
            setting.seed = random.randint(0, 10000000)
            setting.width = 11
            setting.height = 11
            setting.food_spawn_chance = 15
            setting.minimum_food = 1
            result = start_duel_game(client1, client2, setting)

            # ゲーム終了時の自身のスコアを学習
            reward = self.calculate_result_reward(result["result"], result["turn"], result["kill"], result["cause"], self.last_opponent_stack)
            self.total_reward += reward
            last_state_tensor = self.last_you["state"]
            last_action = self.last_you["action"]
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            self.memory.push(last_state_tensor[0], last_state_tensor[1], last_action, reward, last_state_tensor[0], last_state_tensor[1], True)

            # ゲーム終了時の相手のスコアを学習
            res = "draw"
            if result["result"] == "win":
                res = "lose"
            elif result["result"] == "lose":
                res = "win"
            reward = self.calculate_result_reward(res, result["turn"], result["cause"], result["kill"], self.last_you_stack)
            last_state_tensor = self.last_opponent["state"]
            last_action = self.last_opponent["action"]
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            self.memory.push(last_state_tensor[0], last_state_tensor[1], last_action, reward, last_state_tensor[0], last_state_tensor[1], True)

            self.optimize_model()

            # ターゲットネットワークの更新
            if (self.steps_done > START_STEP) and (episode % TARGET_UPDATE == 0):
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (self.steps_done > START_STEP) and (episode % SAVE_STEP == 0):
                evaluator.model.save(f"./checkpoint.pth")
                # turnsをcsvに保存
                with open("./turns.csv", "w") as f:
                    f.write("episode,turn,result,reward\n")
                    for i, res in enumerate(turns):
                        t,r,tr = res
                        f.write(f"{i},{t},{r},{tr}\n")

            print(f"Episode {episode}: Turn{result["turn"]}, Total Reward: {self.total_reward}")
            turns.append([result["turn"], result["cause"], self.total_reward])
            if self.cancel_token.is_cancellation_requested():
                break
        self.finalize()
        self.policy_net.eval()
        buffer = io.BytesIO()
        torch.save({
            'model': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'target': self.target_net.state_dict(),
            'memory': self.memory.memory,
            'episode': episode
        }, buffer)
        return buffer.getvalue()


    def start(self, task):
        data = None
        episode = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        # self.policy_net.load_state_dict(torch.load("C:\\Users\\tokuh\\IntelliJIDEAProjects\\PBL-team4\\duel\\trainer\\checkpoint.pth"))
        if task['baseModelId'] is not None:
            data = self.load_model(task['baseModelId'])
            self.policy_net.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            if 'target' in data:
                self.target_net.load_state_dict(data['target'])
            if 'memory' in data:
                self.memory.memory = data['memory']
            if 'episode' in data:
                episode = data['episode']

        return self.task_reinforced(episode)

def train(api_url: str, logger:Logger, cache_all:bool):
    if not file_exists_and_not_empty("client.json"):
        logger.debug("client.json is empty. Register client first.")
        if not file_exists_and_not_empty("user.txt"):
            logger.error("user.txt is empty. Please create user.txt and write your user id.")
            return
        with open("user.txt", "r") as f:
            user_id = f.read().strip()
        client = ApiClientImpl(api_url=api_url, secret_key="").register_client(user_id)
        logger.debug(f"Registered as {client['id']}")
        with open("client.json", "w") as f:
            f.write(json.dumps(client))

    with open("client.json", "r") as f:
        s = f.read()
    if s:
        client = json.loads(s)
    logger.info(f"Start training as {client["id"]}")
    api_client = ApiClientImpl(api_url=api_url, secret_key=client["secret"], cache_all=cache_all)

    def sig_handler(signum, frame) -> None:
        nonlocal logger
        logger.info("Received signal")
        logger.info("Canceling training...")
        sys.exit(1)

    #期限が近いた時に呼ばれるコールバック
    def timeout_callback(to_id) -> None:
        nonlocal deadline_timer, cancel_token, api_client, logger, deadline
        resp = api_client.refresh_assignment(to_id) #タスクの期限を更新
        #期限が更新できなかった場合, タスクをキャンセル
        if resp is None:
            cancel_token.request_cancel()
            logger.debug(f"Failed to refresh assignment({to_id}). Task is canceled.")
            return
        deadline = datetime.fromtimestamp(resp['deadline']/1000)

        #タイマーを再設定
        deadline_timer.cancel()
        time_out = (deadline - datetime.now()).total_seconds() - 10
        deadline_timer = threading.Timer(time_out, timeout_callback, args=[to_id])
        deadline_timer.start()
        logger.debug(f"Assignment({to_id}) is refreshed. New deadline is {deadline}")

    #トレーニングを開始する
    signal.signal(signal.SIGINT, sig_handler) #シグナルハンドラを設定

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug(f"Start training on device {device}")

    while True:
        print("--------------------------------------------")
        logger.info("Get next assignment...")
        assignment = api_client.get_assignment()

        #タスクがない場合, 2分待って再度取得
        if assignment is None:
            logger.info("No task is assigned. Wait for 2 minutes and try again.")
            sleep(120)
            continue

        assignment_id = assignment['id']
        task = assignment['task']
        deadline = datetime.fromtimestamp(assignment['deadline']/1000)

        logger.info(f"Received assignment({assignment_id}) [deadline:{deadline}]",)
        logger.info(f"Start training for {task}")

        #タイマーを設定
        timeout = (deadline - datetime.now()).total_seconds() - 10
        deadline_timer = threading.Timer(timeout, timeout_callback, args=[assignment_id])

        try:
            cancel_token = CancelToken()
            deadline_timer.start()

            if task['type'] == 'SUPERVISED': #教師あり学習
                result = SupervisedTrainer(logger, api_client, device, cancel_token).start(task)
            elif task['type'] == 'REINFORCEMENT': #強化学習
                result = ReinforcementTrainer(logger, api_client, device, cancel_token).start(task)
            else:
                raise Exception("Unknown task", task)

            if not cancel_token.is_cancellation_requested():
                api_client.submit_model(assignment_id, int(datetime.now().timestamp()*1000), result)
        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Error occurred during training: {e}")
            api_client.post_error(assignment_id, traceback.format_exc(), version_str)
        finally:
            logger.debug("Timer canceled")
            deadline_timer.cancel()
