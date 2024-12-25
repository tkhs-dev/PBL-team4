import abc
import io
import json
import lzma
import sys,os
import zipfile
from datetime import datetime
from typing import Dict, Any

import requests
import torch
import uuid
from requests import Response

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from duel.sneak.duel_evaluator import EvaluatorModel


class ApiClient(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def register_client(self, name: str) -> Dict | None:
        pass

    @abc.abstractmethod
    def get_assignment(self) -> Dict | None:
        pass

    @abc.abstractmethod
    def refresh_assignment(self, assignment_id: str) -> Dict | None:
        pass

    @abc.abstractmethod
    def submit_model(self, assignment_id: str, completed_at: int, model_binary) -> Dict | None:
        pass

    @abc.abstractmethod
    def post_error(self, assignment_id: str, error: str, client_version: str) -> Dict | None:
        pass

    @abc.abstractmethod
    def get_model_bytes(self, model_id: str) -> bytes | None:
        pass

class ApiClientImpl(ApiClient):
    def __init__(self, api_url: str, secret_key: str):
        self.url = api_url
        self.secret_key = secret_key
        self.headers = {'Authorization': f'Bearer {self.secret_key}'}
        self.counter = 0

    def register_client(self, name: str) -> Dict | None:
        response = requests.post(f'{self.url}/clients', json={"user":name})
        response.raise_for_status()
        return response.json()


    def get_assignment(self) -> Dict | None:
        response = self._get('assignments/next')
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response.json()

    def refresh_assignment(self, assignment_id: str) -> Dict | None:
        response = self._post(f'assignments/{assignment_id}/refresh', {})
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response.json()

    def submit_model(self, assignment_id: str, completed_at: int, model_binary) -> Dict | None:
        #モデルを圧縮
        model_binary = lzma.compress(model_binary, preset=9)

        data = {
            'completedAt': completed_at
        }
        files = {
            'json': (None, json.dumps(data), 'application/json'),
            'binary': ('binary_file', model_binary, 'application/octet-stream')
        }
        response = requests.post(f'{self.url}/assignments/{assignment_id}/register', files=files, headers=self.headers)
        response.raise_for_status()
        resp = response.json()
        # キャッシュフォルダに保存
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        with open(f"./cache/{resp['modelId']}", "wb") as f:
            f.write(model_binary)
        return resp

    def post_error(self, assignment_id: str, error: str, client_version: str) -> Dict | None:
        data = {
            'stackTrace': error,
            'clientVersion': client_version
        }
        response = self._post(f'assignments/{assignment_id}/error', data)
        response.raise_for_status()
        return response.json()

    def get_model_bytes(self, model_id: str) -> bytes | None:
        #キャッシュフォルダを検索
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        if os.path.exists(f"./cache/{model_id}"):
            with open(f"./cache/{model_id}", "rb") as f:
                data = f.read()
        else:
            data = requests.get(f'{self.url}/static/models/{model_id}', headers=self.headers)
            data.raise_for_status()
            data = data.content
        # zip解凍
        data = lzma.decompress(data)
        return data

    def _get(self, endpoint: str) -> Response:
        return requests.get(f'{self.url}/{endpoint}', headers=self.headers)

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Response:
        return requests.post(f'{self.url}/{endpoint}', json=data, headers=self.headers)

timeout = 20
class TestApiClient:
    def __init__(self):
        self.secret_key = "test"
        self.repo = {
            'assignment':[]
        }

    def get_assignment(self) -> Dict | None:
        assignment = self.repo.get('assignment')
        filtered = list(filter(lambda x: x['client'] == self.secret_key, assignment))
        if not filtered:
            task = {
                'id': str(uuid.uuid4()),
                'assigned_at': int(datetime.now().timestamp()),
                'client': self.secret_key,
                'deadline': int(datetime.now().timestamp()) + timeout,
                'status': 'PROCESSING',
                'status_changed_at': int(datetime.now().timestamp()),
                'task': {
                    'id': str(uuid.uuid4()),
                    'completed': False,
                    'type': 'supervised',
                    'base_model_id': None,
                    'parameters': {
                        'epochs': 1000,
                        'games':["f3cf360e-b270-4d45-acbd-f7b3337dbeaf","adcc5f77-d9e1-407b-898b-511d49fbfb62","e5c01d03-477d-4568-ad46-b1cb72c266c6","bdc27737-7b01-4f94-a278-d9035ef01375","26e3b70d-e570-46da-85b5-c68d9f947e92","b02e6089-25af-4feb-babb-c053cbd7c0ee","4b004d35-e8b0-48ca-9b34-5c57b571532b","8aab7cea-d020-4ffc-bebe-007957544a0b"],
                        'player_id': 'yannikm'
                    }
                },
            }
            assignment.append(task)
            return task
        return filtered[0]

    def refresh_assignment(self, assignment_id: str) -> Dict | None:
        assignment = self.repo.get('assignment')
        filtered = list(filter(lambda x: x['id'] == assignment_id, assignment))
        if not filtered:
            return None
        task = filtered[0]
        if task['deadline'] < int(datetime.now().timestamp()):
            task['status'] = 'TIMEOUT'
            return None
        task['deadline'] = int(datetime.now().timestamp()) + timeout
        return task

    def submit_model(self, assignment_id: str, completed_at: int, model_binary) -> Dict | None:
        assignment = self.repo.get('assignment')
        filtered = list(filter(lambda x: x['id'] == assignment_id, assignment))
        if not filtered:
            return None
        task = filtered[0]
        task['status'] = 'COMPLETED'
        task['status_changed_at'] = int(datetime.now().timestamp())
        task['completed_at'] = completed_at
        buffer = io.BytesIO(model_binary)
        model = EvaluatorModel()
        model.load_state_dict(torch.load(buffer, weights_only=True))
        model.save(f"{task['task']['id']}.pt")
        return task

    def get_model(self, model_id: str) -> EvaluatorModel | None:
        pass