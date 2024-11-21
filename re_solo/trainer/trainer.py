import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch import optim, nn

from re_solo.sneak.evaluator import Evaluator
from re_solo.sneak.player import AIPlayer
from re_solo.trainer.rules import Client, start_solo_game, GameSettings
from shared.rule import Direction

# ハイパーパラメータ
GAMMA = 0.99  # 割引率
LR = 0.001    # 学習率
BATCH_SIZE = 4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 500
MEMORY_CAPACITY = 10000
TARGET_UPDATE = 10
BOARD_SIZE = 11  # 11x11の盤面
NUM_CHANNELS = 6  # 入力チャネル（自身の頭、体、敵の頭など）

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, board_state, game_state, action, reward, next_board_state, next_game_state, done):
        self.memory.append((board_state, game_state, action, reward, next_board_state, next_game_state, done))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ε-greedyポリシー
def select_action(state, policy_net, steps_done, n_actions, device):
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
    if random.random() > epsilon:
        with torch.no_grad():

            return policy_net(*map(lambda x:x.unsqueeze(0),state)).argmax(dim=1).item()  # 最適な行動
    else:
        return random.randrange(n_actions)  # ランダム行動

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = Evaluator().model.to(device)
target_net = Evaluator().model.to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

# 学習ステップ
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    board_state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
    game_state_batch = torch.tensor(np.array(batch[1]), dtype=torch.float32).to(device)

    action_batch = torch.tensor(batch[2]).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[3], dtype=torch.float32).unsqueeze(1).to(device)
    next_board_state_batch = torch.tensor(np.array(batch[4]), dtype=torch.float32).to(device)
    next_game_state_batch = torch.tensor(np.array(batch[5]), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch[6], dtype=torch.float32).unsqueeze(1).to(device)

    # Q値を計算
    state_action_values = policy_net(board_state_batch,game_state_batch).gather(1, action_batch)
    next_state_values = target_net(next_board_state_batch,next_game_state_batch).max(1)[0].detach().unsqueeze(1)
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # 損失関数
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)

    # ネットワーク更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

state_tensor = None #[board_tensor, game_tensor]
total_reward = 0

def start_callback(game_state):
    global total_reward
    total_reward = 0
    return 0

last_action = None

def move_callback(game_state, player):
    global state_tensor, total_reward, steps_done, last_action, memory, policy_net
    next_state_tensor = Evaluator.get_input_tensor(game_state)
    if(game_state["turn"] > 0):
        reward = 1
        if game_state["you"]["health"] == 100:
            reward += 50
        total_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32).to(device)
        memory.push(state_tensor[0], state_tensor[1], last_action, reward, next_state_tensor[0], next_state_tensor[1], False)
        optimize_model()
    state_tensor = next_state_tensor
    action = select_action(state_tensor, policy_net, steps_done, 4, device)
    last_action = action
    steps_done += 1
    return Direction.index(action)

def end_callback(game_state):
    global total_reward, state_tensor, last_action, memory
    next_state_tensor = Evaluator.get_input_tensor(game_state)
    reward = 0
    head = game_state["you"]["head"]
    if 0<=head["x"]<6 and 0<=head["y"]<6:
        reward = -50
    if game_state["you"]["health"] == 0:
        reward -= 5
    total_reward += reward
    reward = torch.tensor([reward], dtype=torch.float32).to(device)
    memory.push(state_tensor[0], state_tensor[1], last_action, reward, next_state_tensor[0], next_state_tensor[1], True)
    optimize_model()
    print(f"total reward: {total_reward}")
    return 0

def train():
    if not os.path.exists("../pth/"):
        os.makedirs("../pth/")
    os.mkdir("../pth/"+str(int(time.time())))
    path = "../pth/"+str(int(time.time()))+"/"

    # トレーニングループ
    num_episodes = 50000
    for episode in range(num_episodes):
        evaluator = Evaluator()
        evaluator.model = policy_net
        player = AIPlayer(evaluator)
        client = Client()
        client.on_start = start_callback
        client.on_move = lambda game_state: move_callback(game_state, player)
        client.on_end = end_callback
        setting = GameSettings()
        setting.seed = random.randint(0, 10000000)
        setting.width = 6
        setting.height = 6
        setting.food_spawn_chance = 0
        setting.minimum_food = 3
        result = start_solo_game(client, setting)
        print(f"Episode {episode}: Turn{result["turn"]}")
        if episode % 2500 == 0:
            evaluator.model.save(path + f"model_{episode}.pth")
        if episode == num_episodes-1:
            evaluator.model.save(path + f"model_final.pth")
