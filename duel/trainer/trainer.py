import signal
import sys
import threading
import os
import io
from datetime import datetime
from time import sleep

from logging import getLogger, DEBUG, StreamHandler, Formatter

import torch
from torch.utils.data import Dataset

from api_client import TestApiClient, ApiClientImpl
from duel.sneak.duel_evaluator import Evaluator, EvaluatorModel
from game_downloader import GameDownloader
from shared.rule import Direction

# ニューラルネットワークの学習用データセット
class DuelDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = list(map(lambda x:x,Evaluator.get_input_tensor(self.data[idx]['game_state'])))
        target = self.data[idx]['action']
        if target == Direction.UP:
            target = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        elif target == Direction.DOWN:
            target = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
        elif target == Direction.LEFT:
            target = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
        elif target == Direction.RIGHT:
            target = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        return input_tensor, target

class Trainer:
    def __init__(self, logger=None):
        self.logger = logger
        self.timer = None
        self.api_client = None
        self.cancel = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #シグナルハンドラ
    def sig_handler(self,signum, frame) -> None:
        self.logger.info("Received signal")
        self.logger.info("Canceling training...")
        sys.exit(1)

    #モデルをロードする. キャッシュがあればそれを使う.
    def load_model(self, model_id: str) -> EvaluatorModel | None:
        #キャッシュフォルダを検索
        if os.path.exists(f"./cache/{model_id}.pt"):
            model = EvaluatorModel()
            model.load(f"./cache/{model_id}.pt")
            return model

        model = self.api_client.get_model(model_id)
        if model is None:
            self.logger.error(f"Failed to get model({model_id})")
            return None
        model.save(f"./cache/{model_id}.pt") #キャッシュに保存
        return model

    def task_supervised(self, base_model:EvaluatorModel, task, batch_size = 32) -> bytes | None:
        print("Supervised training started")
        games = task['parameters']['games']
        downloader = GameDownloader()
        datas = []
        for game in games:
            #すべてのゲームについてデータをダウンロードし, 最後のターン以外をdatasに追加
            datas += downloader.download_data(game,task['parameters']['player_id'])[:-1]

        dataset = DuelDataset(datas)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = base_model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epoch = task['parameters']['epochs']

        #学習ループ
        for e in range(epoch):
            if self.cancel:
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

            self.logger.debug(f"Epoch {e+1}/{epoch} Loss: {total_loss/len(dataloader)}")
        #学習結果をバイナリに変換して返す
        model.eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.getvalue()

    #期限が近いた時に呼ばれるコールバック
    def timeout_callback(self, assignment_id) -> None:
        resp = self.api_client.refresh_assignment(assignment_id) #タスクの期限を更新
        #期限が更新できなかった場合, タスクをキャンセル
        if resp is None:
            self.cancel = True
            self.logger.debug(f"Failed to refresh assignment({assignment_id}). Task is canceled.")
            return
        deadline = datetime.fromtimestamp(resp['deadline'])

        #タイマーを再設定
        self.timer.cancel()
        time_out = (deadline - datetime.now()).total_seconds() - 10
        self.timer = threading.Timer(time_out, self.timeout_callback, args=[assignment_id])
        self.timer.start()
        self.logger.debug(f"Assignment({assignment_id}) is refreshed. New deadline is {deadline}")

    def start(self, api_client):
        self.api_client = api_client
        signal.signal(signal.SIGINT, self.sig_handler) #シグナルハンドラを設定

        self.logger.debug(f"Start training on device {self.device}")

        while True:
            print("--------------------------------------------")
            self.logger.info("Get next assignment...")
            assignment = self.api_client.get_assignment()

            #タスクがない場合, 2分待って再度取得
            if assignment is None:
                self.logger.info("No task is assigned. Wait for 2 minutes and try again.")
                sleep(120)
                continue

            assignment_id = assignment['id']
            task = assignment['task']
            deadline = datetime.fromtimestamp(assignment['deadline'])

            self.logger.info(f"Received assignment({assignment_id}) [deadline:{deadline}]",)
            self.logger.info(f"Start training for {task}")

            #タイマーを設定
            timeout = (deadline - datetime.now()).total_seconds() - 10
            self.timer = threading.Timer(timeout, self.timeout_callback, args=[assignment_id])

            try:
                self.cancel = False
                self.timer.start()

                #ベースモデルが指定されている場合はロード, 指定されていない新規作成
                if task['base_model_id'] is None:
                    model = EvaluatorModel()
                else:
                    model = self.load_model(task['base_model_id'])
                    if model is None:
                        raise Exception(f"Failed to load model({task['base_model_id']})")

                if task['type'] == 'supervised': #教師あり学習
                    result = self.task_supervised(model, task)
                    if not self.cancel:
                        self.api_client.submit_model(assignment_id, int(datetime.now().timestamp()), result)
                else:
                    raise Exception("Unknown task", task)
            finally:
                self.logger.debug("Timer canceled")
                self.timer.cancel()

def train():
    logger = getLogger("Trainer")
    lvl = DEBUG
    ch = StreamHandler(stream=sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.setLevel(lvl)
    logger.addHandler(ch)
    with open("./secret.txt") as f:
        s = f.read()
    if not s:
        raise Exception("secret.txt is empty")
    logger.debug("Loaded secret key")
    api_client = ApiClientImpl(api_url='http://localhost:8080', secret_key=s)
    # api_client = TestApiClient()
    Trainer(logger).start(api_client)
