from typing import List, Optional, Union
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

from cfc import CFC, model_params, encode_from_json

app = FastAPI()


class ArtifactPredictor:
    def __init__(self, ckpt_path: str):
        self.cfc = CFC.load_from_checkpoint(ckpt_path)
        self.cfc.eval()
        self.event_buffer = []
        self.current_candidates = []

        @app.post("/add_events")
        async def add_events(events: List[dict]):  # 使用明确的Event模型
            self._add_events(events)
            return {"status": "ok"}

        @app.get("/predict")
        async def predict():
            return self._predict()

        @app.get("/check")
        async def check():
            return {"status": "ok"}

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

    def _predict(self) -> list:
        if not self.current_candidates:
            return []

        seq = encode_from_json(self.event_buffer, is_inference=True)
        ret = self.cfc.inference(seq, self.current_candidates, k=model_params['k'])
        print('predict:', ret)
        return ret

    def launch_server(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    predictor = ArtifactPredictor('cfc/ckpt/cfc/version_1/cfc-best.ckpt')
    predictor.launch_server()