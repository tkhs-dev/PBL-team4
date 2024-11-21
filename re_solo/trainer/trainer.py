import random
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
BATCH_SIZE = 64
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

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def select_action(state, player, steps_done):
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
    if random.random() > epsilon:
        with torch.no_grad():
            return player.on_move(state)
    else:
        return random.choice(list(Direction))

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
    # print(batch)
    state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
    next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

    # Q値を計算
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
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
    global state_tensor, total_reward, steps_done, last_state, last_action, memory
    next_state_tensor = Evaluator.get_input_tensor(game_state)
    if(game_state["turn"] > 0):
        reward = 1
        if game_state["you"]["health"] == 100:
            reward += 50
        total_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32).to(device)
        memory.push(state_tensor, last_action, reward, next_state_tensor, False)
        optimize_model()
    state_tensor = next_state_tensor
    action = select_action(game_state, player, steps_done)
    last_action = action
    steps_done += 1
    return action

def end_callback(game_state):
    global total_reward, state_tensor, last_action, memory
    next_state_tensor = Evaluator.get_input_tensor(game_state)
    reward = torch.tensor([-1], dtype=torch.float32).to(device)
    memory.push(state_tensor, last_action, reward, next_state_tensor, True)
    optimize_model()
    print(f"total reward: {total_reward}")
    return 0

def train():
    # トレーニングループ
    num_episodes = 20
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