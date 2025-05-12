import requests
import time
from cfc import concat_json_from_folder


class ArtifactPredictorClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url

    def add_events(self, events: list):
        """发送事件数据到服务器"""
        response = requests.post(
            f"{self.server_url}/add_events",
            json=events,
            headers={'Content-Type': 'application/json'}
        )
        return response.json()

    def predict(self):
        """获取预测结果"""
        response = requests.get(f"{self.server_url}/predict")
        return response.json()


if __name__ == '__main__':
    # 1. 初始化客户端
    client = ArtifactPredictorClient()

    # 2. 加载测试数据
    print("Loading test data...")
    input_json = concat_json_from_folder('./dataset_raw')
    print(f"Loaded {len(input_json)} events")

    # 3. 发送测试数据到服务器
    client.add_events(input_json)

    # 4. 最终预测
    print("\nFinal prediction:")
    print(client.predict())