import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import os


class CFC:
    def __init__(self, in_features, out_features, units):
        self.wiring = AutoNCP(units, out_features)
        self.model = CfC(in_features, self.wiring, batch_first=True)

    def draw_structure(self, save_dir='visualizations'):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 创建保存目录
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

    def inference(self, x):
        with torch.no_grad():
            return self.model.forward(x)[0]


if __name__ == '__main__':
    # 创建模型
    cfc = CFC(2, 1, 16)
    
    # 生成并保存网络结构图
    cfc.draw_structure()
    
    # 测试推理
    x = torch.randn(1, 100, 2)  # batch_size, seq_len, in_features
    y = cfc.inference(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")