import json
import signal
import sys
import threading
import os
import io
import traceback
from datetime import datetime
from time import sleep

from logging import Logger

import torch
from early_stopping_pytorch import EarlyStopping
from torch.utils.data import Dataset

from api_client import TestApiClient, ApiClientImpl
from duel.sneak.duel_evaluator import Evaluator, EvaluatorModel
from duel.trainer.api_client import ApiClient
from duel.trainer.common import file_exists_and_not_empty, get_version_str
from game_downloader import GameDownloader
from shared import rule
from shared.rule import Direction, TurnResult

version = "1.0.0"
version_str = get_version_str(version)

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
            self.datas.append(turn)


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        input_tensor = self.datas[idx]['input']
        target = self.datas[idx]['target']
        return input_tensor, target

class Trainer:
    def __init__(self, logger=None):
        self.logger = logger
        self.timer = None
        self.api_client:ApiClient
        self.cancel = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #シグナルハンドラ
    def sig_handler(self,signum, frame) -> None:
        self.logger.info("Received signal")
        self.logger.info("Canceling training...")
        sys.exit(1)

    #モデルをロードする. キャッシュがあればそれを使う.
    def load_model(self, model_id: str):
        data = self.api_client.get_model_bytes(model_id)
        data = torch.load(io.BytesIO(data), map_location=self.device, weights_only=True)
        model = EvaluatorModel()
        model.load_state_dict(data['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(data['optimizer'])
        return model,optimizer

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
                data = downloader.download_data(game,task['parameters']['player_id'])
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

    #期限が近いた時に呼ばれるコールバック
    def timeout_callback(self, assignment_id) -> None:
        resp = self.api_client.refresh_assignment(assignment_id) #タスクの期限を更新
        #期限が更新できなかった場合, タスクをキャンセル
        if resp is None:
            self.cancel = True
            self.logger.debug(f"Failed to refresh assignment({assignment_id}). Task is canceled.")
            return
        deadline = datetime.fromtimestamp(resp['deadline']/1000)

        #タイマーを再設定
        self.timer.cancel()
        time_out = (deadline - datetime.now()).total_seconds() - 10
        self.timer = threading.Timer(time_out, self.timeout_callback, args=[assignment_id])
        self.timer.start()
        self.logger.debug(f"Assignment({assignment_id}) is refreshed. New deadline is {deadline}")

    def start(self, api_client):
        self.api_client = api_client
        signal.signal(signal.SIGINT, self.sig_handler) #シグナルハンドラを設定

        if not os.path.exists("./cache/games"):
            os.makedirs("./cache/games")

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
            deadline = datetime.fromtimestamp(assignment['deadline']/1000)

            self.logger.info(f"Received assignment({assignment_id}) [deadline:{deadline}]",)
            self.logger.info(f"Start training for {task}")

            #タイマーを設定
            timeout = (deadline - datetime.now()).total_seconds() - 10
            self.timer = threading.Timer(timeout, self.timeout_callback, args=[assignment_id])

            try:
                self.cancel = False
                self.timer.start()

                #ベースモデルが指定されている場合はロード, 指定されていない新規作成
                if task['baseModelId'] is None:
                    model = EvaluatorModel()
                    optimizer = None
                else:
                    model,optimizer = self.load_model(task['baseModelId'])
                    if model is None:
                        raise Exception(f"Failed to load model({task['baseModelId']})")

                if task['type'] == 'SUPERVISED': #教師あり学習
                    result = self.task_supervised(model, optimizer,task)
                    if not self.cancel:
                        self.api_client.submit_model(assignment_id, int(datetime.now().timestamp()), result)
                else:
                    raise Exception("Unknown task", task)
            except Exception as e:
                print(traceback.format_exc())
                self.logger.error(f"Error occurred during training: {e}")
                self.api_client.post_error(assignment_id, str(e), version_str)
            finally:
                self.logger.debug("Timer canceled")
                self.timer.cancel()

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
    # api_client = TestApiClient()
    Trainer(logger).start(api_client)
