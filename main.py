import datetime
import multiprocessing
import os
import shutil
from typing import List, Dict

import uvicorn
from fastapi import FastAPI

from cfc import CFC, model_params, encode_from_json, train

app = FastAPI()

class ArtifactPredictor:
    def __init__(self, ckpt_dir: str):
        self.cfc = None
        self.ckpt_dir = ckpt_dir
        self.event_buffer = []
        self.current_candidates = []
        self.state = {
            'is_predicting': multiprocessing.Value('b', False),
            'is_updating': multiprocessing.Value('b', False),
            'need_change_model': multiprocessing.Value('b', False),
            'is_changing_model': multiprocessing.Value('b', False),
        }
        self._load_model()

        @app.post("/add_events")
        async def add_events(events: List[dict]):
            self._add_events(events)

        @app.get("/predict")
        async def predict():
            return self._predict()

        @app.get("/check")
        async def check():
            return {"status": "ok"}

        @app.post('/update')
        async def update(raw_data_path: Dict[str, str]):
            self._update(raw_data_path['raw_data_path'])

    def _load_model(self):
        if self.state['is_predicting'].value or self.state['is_changing_model'].value:
            print('is predicting or changing model')
            return
        self.state['is_changing_model'].value = True
        self.state['need_change_model'].value = False

        # 获取self.ckpt_dir下最新模型，模型名是datetime的格式，如20220101123456
        latest_ckpt = max(os.listdir(self.ckpt_dir), key=lambda x: datetime.datetime.strptime(x, '%Y%m%d%H%M%S.ckpt'))
        # 删除其他旧模型
        for ckpt in os.listdir(self.ckpt_dir):
            if ckpt != latest_ckpt:
                os.remove(os.path.join(self.ckpt_dir, ckpt))
        # 加载最新模型
        self.cfc = CFC.load_from_checkpoint(os.path.join(self.ckpt_dir, latest_ckpt))
        self.cfc.eval()
        print('loaded model: ', latest_ckpt)
        self.state['is_changing_model'].value = False

    def _train_proc(self, dataset_dir: str, date_filter: datetime.datetime):
        train(dataset_dir, date_filter)
        self.state['need_change_model'].value = True
        self.state['is_updating'].value = False

    def _update(self, raw_data_path: str):
        if self.state['is_updating'].value:
            print('is already updating')
            return

        self.state['is_updating'].value = True
        os.makedirs('dataset_raw', exist_ok=True)
        shutil.copy(raw_data_path, 'dataset_raw')
        # 使用至多7天前的数据进行训练
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=30)

        p = multiprocessing.Process(target=self._train_proc, args=('dataset_raw', yesterday))
        p.start()

    def _add_events(self, events: List[dict]):  # 类型注解更新为List[Event]
        self.event_buffer.extend(events)
        print('add_events:', events)
        # 获取最新的候选集
        for event in reversed(self.event_buffer):
            if event['candidates'] is not None:
                self.current_candidates = event['candidates']
                break
        # 保持缓冲区长度
        self.event_buffer = self.event_buffer[-model_params['max_seq_len']:]

        if self.state['need_change_model'].value:
            self._load_model()

    def _predict(self) -> list:
        if not self.current_candidates:
            print('no candidates')
            return []
        if not self.cfc:
            print('no model')
            return []
        if self.state['is_predicting'].value:
            print('is already predicting')
            return []
        if self.state['is_changing_model'].value:
            print('is changing model')
            return []

        self.state['is_predicting'].value = True
        seq = encode_from_json(self.event_buffer, is_inference=True)
        ret = self.cfc.inference(seq, self.current_candidates, k=model_params['k'])
        print('predict:', ret)
        self.state['is_predicting'].value = False
        return ret

    def launch_server(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    predictor = ArtifactPredictor('./ckpt')
    predictor.launch_server()
