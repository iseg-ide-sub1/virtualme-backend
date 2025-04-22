import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP

try:
    from .config import model_params
except ImportError:
    from config import model_params


class CFC:
    def __init__(self):
        in_features = model_params['event_type_embedding_dim'] + model_params['feedback_dim'] + model_params[
            'artifact_embedding_dim']
        units = model_params['units']
        out_features = model_params['artifact_embedding_dim'] * model_params['pred_len']

        self.wiring = AutoNCP(units, out_features)
        self.model = CfC(in_features, self.wiring, batch_first=True)

    def draw_structure(self):
        # 创建保存目录
        save_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'network_structure.png')

        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存
        print(f"Network structure visualization saved to: {save_path}")

    def inference(self, x, t):
        with torch.no_grad():
            return self.model.forward(x, timespans=t)[0]


if __name__ == '__main__':
    # 创建模型
    cfc = CFC()

    # 生成并保存网络结构图
    cfc.draw_structure()

    # 测试推理
    in_features = model_params['event_type_embedding_dim'] + model_params['feedback_dim'] + model_params[
        'artifact_embedding_dim']
    x = torch.randn(1, 100, in_features)  # batch_size, seq_len, in_features
    # 假设t是时间戳序列，生成长度为100的从小到大排列的序列
    t = torch.arange(100)
    t = t.expand(1, 100)
    y = cfc.inference(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
