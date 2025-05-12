from fastapi import FastAPI
import uvicorn
from typing import List, Dict, Any
from cfc import CFC, model_params, encode_from_json

app = FastAPI()


class ArtifactPredictor:
    def __init__(self, ckpt_path: str):
        self.cfc = CFC.load_from_checkpoint(ckpt_path)
        self.cfc.eval()
        self.event_buffer = []
        self.current_candidates = []

        @app.post("/add_events")
        async def add_events(events: List[Dict[str, Any]]):
            self._add_events(events)
            return {"status": "ok"}

        @app.get("/predict")
        async def predict():
            return self._predict()

    def _add_events(self, events: List[Dict[str, Any]]):
        self.event_buffer.extend(events)
        # 获取最新的候选集
        for event in reversed(self.event_buffer):
            if 'candidates' in event:
                self.current_candidates = event['candidates']
                break
        # 保持缓冲区长度
        self.event_buffer = self.event_buffer[-model_params['max_seq_len']:]

    def _predict(self) -> List[Dict[str, Any]]:
        if not self.current_candidates:
            return []

        seq = encode_from_json(self.event_buffer, is_inference=True)
        return self.cfc.inference(seq, self.current_candidates, k=model_params['k'])

    def launch_server(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    predictor = ArtifactPredictor('cfc/ckpt/cfc/version_1/cfc-best.ckpt')
    predictor.launch_server()
