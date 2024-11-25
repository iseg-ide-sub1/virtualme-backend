import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class CFC:
    def __init__(self, in_features, out_features, units):
        self.wiring = AutoNCP(units, out_features)
        self.model = CfC(in_features, self.wiring, batch_first=True)

    def draw_structure(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("white")
        plt.figure(figsize=(6, 4))
        legend_handles = self.wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()

    def inference(self, x):
        with torch.no_grad():
            return self.model.forward(x)[0]


if __name__ == '__main__':
    cfc = CFC(2, 1, 16)
    cfc.draw_structure()