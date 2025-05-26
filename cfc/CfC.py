import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.nn import MSELoss

try:
    from .config import model_params, train_params
except ImportError:
    from config import model_params, train_params


class CFC(pl.LightningModule):
    def __init__(self, lr=0.001, decay_lr=0.97, weight_decay=1e-4):
        super().__init__()
        in_features = model_params['event_type_embedding_dim'] + model_params['feedback_dim'] + model_params[
            'artifact_embedding_dim']
        units = model_params['units']
        out_features = model_params['artifact_embedding_dim']

        self.wiring = AutoNCP(units, out_features)
        self.model = CfC(in_features, self.wiring, batch_first=True, return_sequences=False)
        self.lr = lr
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.MSELoss(reduction='none')
        self.mean_acc_size = train_params['mean_acc_size']
        self.last_train_acc_scores = []
        self.last_val_acc_scores = []

    def forward(self, x, time):
        y = self.model(x, timespans=time)
        y = y[0] if isinstance(y, tuple) else y
        return y

    def retrieve_candidate_embed(self, pred_embed: torch.Tensor, candidate_embeds: torch.Tensor, k: int = 1) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        根据预测的嵌入向量，从候选嵌入向量中检索最相似的k个。
        现在处理的是单个时间步的预测和候选。
        Args:
            pred_embed: [B, D] tensor of predicted embeddings (最后一个时间步的预测)
            candidate_embeds: [B, C, D] tensor of candidate embeddings (最后一个时间步的候选)
            k: number of top candidates to retrieve (default: 1)

        Returns:
            tuple containing:
            - match_embeds: [B, k, D] tensor of matched candidate embeddings
            - match_indices: [B, k] tensor of candidate indices (relative to each item's candidate set)
        """
        B, D = pred_embed.shape
        _, C, _ = candidate_embeds.shape  # B, C, D

        mse_loss_fn = nn.MSELoss(reduction='none')
        # 自动广播，计算每个候选与预测的距离
        mse_distances_all_features = mse_loss_fn(pred_embed.unsqueeze(1), candidate_embeds)  # [B, C, D]
        mse_distances = mse_distances_all_features.mean(dim=-1)  # [B, C], 在特征维度D上取平均得到距离
        # 为每个批次项选择 top-k 最近的候选
        _, match_indices = torch.topk(mse_distances, k, dim=1, largest=False)  # [B, k]
        # 收集匹配的嵌入向量
        expanded_indices = match_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, k, D]
        match_embeds = torch.gather(candidate_embeds, 1, expanded_indices)  # [B, k, D]
        return match_embeds, match_indices

    def acc_fn(self, artifact_pred_matched, label_embed):
        # 计算匹配矩阵 [B, 1, k]
        match_matrix = torch.all(
            torch.Tensor(artifact_pred_matched == label_embed),
            dim=-1
        )
        # 直接计算平均匹配率
        return match_matrix.float().mean()

    def prepare_batch(self, batch):
        time, event_type_embed, feedback_embed, artifact_embed, candidate_embed, label_embed = batch

        t_elapsed = time[:, 1:] - time[:, :-1]
        t_fill = torch.zeros(time.size(0), 1, device=time.device)
        time_processed = torch.cat((t_fill, t_elapsed), dim=1)
        time_processed = time_processed * model_params['tau']

        x = torch.cat((event_type_embed, feedback_embed, artifact_embed), dim=2) # 输入序列保持不变

        label_embed_last_step = label_embed[:, -1, :] # [B, D]
        candidate_embed_last_step = candidate_embed[:, -1, :, :] # [B, C, D]

        return time_processed, x, candidate_embed_last_step, label_embed_last_step

    def prepare_inference_input(self, input_seq: tuple):
        time, event_type_embed, feedback_embed, artifact_embed, candidate_embed, _, _ = input_seq

        time = time.unsqueeze(0)  # [1, T]
        event_type_embed = event_type_embed.unsqueeze(0)  # [1, T, D_event]
        feedback_embed = feedback_embed.unsqueeze(0)  # [1, T, D_feedback]
        artifact_embed = artifact_embed.unsqueeze(0)  # [1, T, D_artifact]

        t_elapsed = time[:, 1:] - time[:, :-1]
        t_fill = torch.zeros(time.size(0), 1, device=time.device)
        time_processed = torch.cat((t_fill, t_elapsed), dim=1)
        time_processed = time_processed * model_params['tau']

        x = torch.cat((event_type_embed, feedback_embed, artifact_embed), dim=2)  # [B, T, D_in]

        # 只选择最后一个时间步的候选，并调整形状为 [1, C, D]
        candidate_embed_last_step = candidate_embed[-1, :, :].unsqueeze(0)  # [1, C, D]

        return time_processed, x, candidate_embed_last_step

    def training_step(self, batch, batch_idx):
        time, x, candidate_embed, label_embed = self.prepare_batch(batch)

        artifact_pred = self.forward(x, time)

        loss = self.loss_fn(artifact_pred, label_embed).mean()

        match_embeds, match_indices = self.retrieve_candidate_embed(
            artifact_pred,  # [B, D]
            candidate_embed  # [B, C, D]
        )
        acc = self.acc_fn(match_embeds[:, 0, :], label_embed)
        self.last_train_acc_scores.append(acc.item())
        self.last_train_acc_scores = self.last_train_acc_scores[-self.mean_acc_size:]
        mean_acc = sum(self.last_train_acc_scores) / len(self.last_train_acc_scores)
        self.log_dict({'train_loss': loss, 'train_acc': mean_acc})
        return loss

    def validation_step(self, batch, batch_idx):
        time, x, candidate_embed, label_embed = self.prepare_batch(batch)

        artifact_pred = self.forward(x, time)

        loss = self.loss_fn(artifact_pred, label_embed).mean()

        match_embeds, match_indices = self.retrieve_candidate_embed(
            artifact_pred,  # [B, D]
            candidate_embed  # [B, C, D]
        )
        acc = self.acc_fn(match_embeds[:, 0, :], label_embed)
        self.last_val_acc_scores.append(acc.item())
        self.last_val_acc_scores = self.last_val_acc_scores[-self.mean_acc_size:]
        mean_acc = sum(self.last_val_acc_scores) / len(self.last_val_acc_scores)
        self.log_dict({'val_loss': loss, 'val_acc': mean_acc})
        return loss

    def inference(self, input_seq: tuple, candidates: List[object], k=1) -> List[object]:
        with torch.no_grad():
            if self.loss_fn is None:
                self.loss_fn = MSELoss(reduction='none')

            time, x, candidate_embed_for_last_step = self.prepare_inference_input(input_seq)

            artifact_pred = self.forward(x, time)

            B, _, D_model_in = x.shape  # x 是 [1, T, D_in]
            if B != 1:
                raise ValueError("推理模式下批次大小必须为 1")

            _, match_indices = self.retrieve_candidate_embed(artifact_pred, candidate_embed_for_last_step, k)
            match_indices_list = match_indices[0, :].tolist()  # 从 [1,k] 中获取 k 个索引
            matched_candidates = [candidates[int(i)] for i in match_indices_list if int(i) < len(candidates)]
            return matched_candidates

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.decay_lr
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

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
        print(f"网络结构可视化已保存到: {save_path}")
