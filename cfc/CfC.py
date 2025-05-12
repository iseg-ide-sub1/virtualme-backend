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

try:
    from .config import model_params
except ImportError:
    from config import model_params


class SequenceMultiStepLoss(nn.Module):
    def __init__(self,
                 base: str = 'mse',
                 huber_delta: float = 1.0,
                 weight_time: torch.Tensor = None,  # [T]
                 trend_weight: float = 0.0):
        """
        :param base: 'mse' | 'mae' | 'huber' | 'logcosh'
        :param huber_delta: Huber Loss 中的 δ
        :param weight_time:    [T] 张量，不同时刻的权重；None→等权
        :param trend_weight: 趋势一致性损失的系数（在 T 维度上一阶差分）
        """
        super().__init__()
        self.base = base
        self.delta = huber_delta
        self.w_t = weight_time  # shape [T]
        self.trend_w = trend_weight

        # 基础 Loss 选择
        if base == 'huber':
            self.criterion = nn.SmoothL1Loss(beta=self.delta, reduction='none')
        elif base == 'mae':
            self.criterion = nn.L1Loss(reduction='none')
        elif base == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif base == 'logcosh':
            self.criterion = None
        else:
            raise ValueError(f"Unsupported base loss '{base}'")

    def forward(self, pred: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param pred: Tensor of shape [B, T, D], predictions
        :param labels: Tensor of shape [B, T, D], ground truth labels
        :param mask: Tensor of shape [B, T, 1], 0 for valid, 1 for invalid time steps
        :return: scalar loss
        """
        B, T, D = pred.shape

        # 1. 计算基础误差 → [B, T, D]
        if self.base == 'logcosh':
            diff = pred - labels
            err = torch.log(torch.cosh(diff))  # [B, T, D]
        else:
            err = self.criterion(pred, labels)  # [B, T, D]

        # 2. 按特征维度 D 平均 → [B, T]
        loss_bt = err.mean(dim=2)

        # 3. 定义权重张量
        device = pred.device
        w_t_tensor = self.w_t.view(1, T) if self.w_t is not None else torch.ones(1, T, device=device)

        # 4. 应用权重 → [B, T]
        loss_bt_weighted = loss_bt * w_t_tensor

        # 5. 应用掩码 → [B, T]
        mask_tensor = (1 - mask).view(B, T)  # [B, T], 1 for valid, 0 for invalid
        loss_bt_masked = loss_bt_weighted * mask_tensor

        # 6. 计算基础损失的总和和归一化因子
        total_loss_base = loss_bt_masked.sum()
        normalization_base = (w_t_tensor * mask_tensor).sum()
        loss_base = total_loss_base / (normalization_base + 1e-8)  # 避免除以零

        # 7. 计算趋势一致性损失（如果适用）
        if self.trend_w > 0:
            dp = pred[:, 1:, :] - pred[:, :-1, :]  # [B, T-1, D]
            dy = labels[:, 1:, :] - labels[:, :-1, :]  # [B, T-1, D]
            trend_err = (dp - dy).pow(2).mean(dim=2)  # [B, T-1]
            trend_err_masked = trend_err * mask_tensor[:, :-1]  # [B, T-1]
            total_trend_loss = trend_err_masked.sum()
            normalization_trend = mask_tensor[:, :-1].sum()  # 有效时间步数
            loss_trend = total_trend_loss / (normalization_trend + 1e-8)
        else:
            loss_trend = 0

        # 8. 组合损失
        loss = loss_base + self.trend_w * loss_trend
        return loss


class CFC(pl.LightningModule):
    def __init__(self, lr=0.001, decay_lr=0.97, weight_decay=1e-4, loss_fn=None):
        super().__init__()
        in_features = model_params['event_type_embedding_dim'] + model_params['feedback_dim'] + model_params[
            'artifact_embedding_dim']
        units = model_params['units']
        out_features = model_params['artifact_embedding_dim']

        self.wiring = AutoNCP(units, out_features)
        self.model = CfC(in_features, self.wiring, batch_first=True)
        self.lr = lr
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.save_hyperparameters()

    def forward(self, x, time):
        y = self.model(x, timespans=time)
        y = y[0] if isinstance(y, tuple) else y
        return y

    def retrieve_candidate_embed(self, pred_embed, candidate_embeds, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_embed: [B, T, D] tensor of predicted embeddings
            candidate_embeds: [B, T, C, D] tensor of candidate embeddings
            k: number of top candidates to retrieve (default: 1)

        Returns:
            tuple containing:
            - match_embeds: [B, T, k, D] tensor of matched candidate embeddings
            - match_indices: [B, T, k] tensor of candidate indices (relative to each T's candidate set)
        """
        B, T, D = pred_embed.shape
        _, _, C, _ = candidate_embeds.shape

        # Initialize loss func
        mse_loss = self.loss_fn.criterion

        # Reshape tensors for efficient computation
        pred_embeds_flat = pred_embed.view(B * T, D)  # [B*T, D]
        candidate_embeds_flat = candidate_embeds.view(B * T * C, D)  # [B*T*C, D]

        # Compute MSE distances between predictions and candidates
        pred_exp = pred_embeds_flat.unsqueeze(1)  # [B*T, 1, D]
        cand_exp = candidate_embeds_flat.unsqueeze(0)  # [1, B*T*C, D]
        mse_distances = mse_loss(pred_exp.expand(-1, B * T * C, -1),
                                 cand_exp.expand(B * T, -1, -1))  # [B*T, B*T*C, D]
        mse_distances = mse_distances.mean(dim=-1)  # [B*T, B*T*C]

        # Reshape distances to separate B and T dimensions
        mse_distances = mse_distances.view(B * T, B * T * C)  # [B*T, B*T*C]

        # For each B*T, select the top-k closest candidates
        global_indices = torch.zeros(B * T, k, dtype=torch.long, device=pred_embed.device)
        local_indices = torch.zeros(B * T, k, dtype=torch.long, device=pred_embed.device)

        for bt in range(B * T):
            # Get the C candidates for this specific B,T pair
            start_idx = (bt // T) * T * C + (bt % T) * C
            end_idx = start_idx + C
            bt_distances = mse_distances[bt, start_idx:end_idx]  # [C]

            # Get indices of k smallest distances
            _, indices = torch.topk(bt_distances, k, dim=-1, largest=False)
            global_indices[bt] = start_idx + indices
            local_indices[bt] = indices

        # Gather the matching embeddings
        match_embeds_flat = candidate_embeds_flat[global_indices]  # [B*T, k, D]
        match_embeds = match_embeds_flat.view(B, T, k, D)  # [B, T, k, D]
        match_indices = local_indices.view(B, T, k)  # [B, T, k]

        return match_embeds, match_indices

    def acc_fn(self, artifact_pred, label_embed, mask) -> float:
        """
        Args:
            artifact_pred: [B, T, D] tensor of predicted artifacts
            label_embed: [B, T, D] tensor of ground truth labels
            mask: [B, T, 1] tensor where 0=valid, 1=invalid

        Returns:
            float: average match rate across all valid time steps
        """
        # 1. 计算匹配矩阵 [B, T, D]
        match_matrix = (artifact_pred == label_embed)

        # 2. 创建有效掩码 [B, T, D]
        valid_mask = (1 - mask).expand(-1, -1, artifact_pred.shape[2])  # [B,T,D]

        # 3. 计算有效匹配数 [B]
        valid_match_num = (match_matrix * valid_mask).sum(dim=(1, 2))  # [B]

        # 4. 计算有效时间步数 [B]
        valid_time_steps = (1 - mask).sum(dim=1)  # [B,1]

        # 5. 计算总有效比较次数 [B]
        total_valid_comparisons = valid_time_steps * label_embed.shape[-1]  # [B,1]

        # 6. 计算匹配率 [B]
        match_rate = valid_match_num / (total_valid_comparisons.squeeze(1) + 1e-8)  # [B]

        # 7. 返回平均匹配率
        return match_rate.mean()

    def prepare_batch(self, batch):
        time, event_type_embed, feedback_embed, artifact_embed, candidate_embed, label_embed, mask = batch
        # 打印time的类型
        breakpoint()
        t_elapsed = time[:, 1:] - time[:, :-1]
        t_fill = torch.zeros(time.size(0), 1, device=time.device)
        time = torch.cat((t_fill, t_elapsed), dim=1)
        time = time * model_params['tau']

        x = torch.cat((event_type_embed, feedback_embed, artifact_embed), dim=2)

        return time, x, candidate_embed, label_embed, mask

    def prepare_inference_input(self, input_seq: tuple):
        time, event_type_embed, feedback_embed, artifact_embed, candidate_embed, _, _ = input_seq

        time = time.unsqueeze(0)  # [1, T]
        event_type_embed = event_type_embed.unsqueeze(0)  # [1, T, D]
        feedback_embed = feedback_embed.unsqueeze(0)  # [1, T, D]
        artifact_embed = artifact_embed.unsqueeze(0)  # [1, T, D]
        candidate_embed = candidate_embed.unsqueeze(0)  # [1, T, C, D]

        t_elapsed = time[:, 1:] - time[:, :-1]
        t_fill = torch.zeros(time.size(0), 1, device=time.device)
        time = torch.cat((t_fill, t_elapsed), dim=1)
        time = time * model_params['tau']

        x = torch.cat((event_type_embed, feedback_embed, artifact_embed), dim=2)

        return time, x, candidate_embed  # [1, T, D]

    def training_step(self, batch, batch_idx):
        time, x, candidate_embed, label_embed, mask = self.prepare_batch(batch)

        artifact_pred = self.forward(x, time)

        # artifact_preds: [B, T, D]
        # label_embed: [B, T, D]
        # mask: [B, T, 1]
        loss = self.loss_fn(artifact_pred, label_embed, mask)
        self.log('train_loss', loss)

        artifact_pred_matched, _ = self.retrieve_candidate_embed(artifact_pred, candidate_embed)

        acc = self.acc_fn(artifact_pred_matched, label_embed, mask)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        time, x, candidate_embed, label_embed, mask = self.prepare_batch(batch)

        artifact_pred = self.forward(x, time)

        loss = self.loss_fn(artifact_pred, label_embed, mask)
        self.log('val_loss', loss)

        artifact_pred_matched, _ = self.retrieve_candidate_embed(artifact_pred, candidate_embed)

        acc = self.acc_fn(artifact_pred_matched, label_embed, mask)
        self.log('val_acc', acc)

        return loss

    def inference(self, input_seq: tuple, candidates: List[object], k=1) -> List[object]:
        with torch.no_grad():
            if self.loss_fn is None:
                self.loss_fn = SequenceMultiStepLoss()
            time, x, candidate_embed = self.prepare_inference_input(input_seq)

            B, T, D = x.shape
            if B != 1:
                raise ValueError("Batch size must be 1 for inference")
            if T > model_params['max_seq_len']:
                raise ValueError(
                    f"Input sequence length {T} exceeds maximum sequence length {model_params['max_seq_len']}")
            if D != model_params['event_type_embedding_dim'] + model_params['feedback_dim'] + \
                    model_params[
                        'artifact_embedding_dim']:
                raise ValueError(
                    f"Input dimension {D} does not match expected dimension {model_params['event_type_embedding_dim']} + "
                    f"{model_params['feedback_dim']} + "
                    f"{model_params['artifact_embedding_dim']}")
            artifact_pred = self.forward(x, time)
            _, match_indices = self.retrieve_candidate_embed(artifact_pred, candidate_embed, k)

            # 取最后一个时间步的预测结果
            match_indices = match_indices[0, -1, :].tolist()
            matched_candidates = [candidates[int(i)] for i in match_indices if i < len(candidates)]

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
        print(f"Network structure visualization saved to: {save_path}")
