import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class SequenceMultiStepLoss(nn.Module):
    def __init__(self,
                 base: str = 'mse',
                 huber_delta: float = 1.0,
                 weight_horizon: torch.Tensor = None,  # [k]
                 weight_time: torch.Tensor = None,  # [T]
                 trend_weight: float = 0.0):
        """
        :param base: 'mse' | 'mae' | 'huber' | 'logcosh'
        :param huber_delta: Huber Loss 中的 δ
        :param weight_horizon: [k] 张量，不同预测步的权重；None→等权
        :param weight_time:    [T] 张量，不同时刻的权重；None→等权
        :param trend_weight: 趋势一致性损失的系数（在 k 维度上一阶差分）
        """
        super().__init__()
        self.base = base
        self.delta = huber_delta
        self.w_k = weight_horizon  # shape [k]
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

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param preds: Tensor of shape [B, T, k, D], predictions
        :param labels: Tensor of shape [B, T, k, D], ground truth labels
        :param mask: Tensor of shape [B, T, 1], 0 for valid, 1 for invalid time steps
        :return: scalar loss
        """
        B, T, k, D = preds.shape

        # 1. 计算基础误差 → [B, T, k, D]
        if self.base == 'logcosh':
            diff = preds - labels
            err = torch.log(torch.cosh(diff))  # [B, T, k, D]
        else:
            err = self.criterion(preds, labels)  # [B, T, k, D]

        # 2. 按特征维度 D 平均 → [B, T, k]
        loss_bt_k = err.mean(dim=3)

        # 3. 定义权重张量
        device = preds.device
        w_k_tensor = self.w_k.view(1, 1, k) if self.w_k is not None else torch.ones(1, 1, k, device=device)
        w_t_tensor = self.w_t.view(1, T, 1) if self.w_t is not None else torch.ones(1, T, 1, device=device)

        # 4. 应用权重 → [B, T, k]
        loss_bt_k_weighted = loss_bt_k * w_k_tensor * w_t_tensor

        # 5. 应用掩码 → [B, T, k]
        mask_tensor = (1 - mask).view(B, T, 1)  # [B, T, 1], 1 for valid, 0 for invalid
        loss_bt_k_masked = loss_bt_k_weighted * mask_tensor

        # 6. 计算基础损失的总和和归一化因子
        total_loss_base = loss_bt_k_masked.sum()
        normalization_base = (w_k_tensor * w_t_tensor * mask_tensor).sum()
        loss_base = total_loss_base / (normalization_base + 1e-8)  # 避免除以零

        # 7. 计算趋势一致性损失（如果适用）
        if self.trend_w > 0:
            dp = preds[:, :, 1:, :] - preds[:, :, :-1, :]  # [B, T, k-1, D]
            dy = labels[:, :, 1:, :] - labels[:, :, :-1, :]  # [B, T, k-1, D]
            trend_err = (dp - dy).pow(2).mean(dim=3)  # [B, T, k-1]
            trend_err_masked = trend_err * mask_tensor  # 广播到 [B, T, k-1]
            total_trend_loss = trend_err_masked.sum()
            normalization_trend = mask_tensor.sum() * (k - 1)  # 有效时间步数 × (k-1)
            loss_trend = total_trend_loss / (normalization_trend + 1e-8)
        else:
            loss_trend = 0

        # 8. 组合损失
        loss = loss_base + self.trend_w * loss_trend
        return loss


class Learner(pl.LightningModule):
    def __init__(self, model, lr=0.001, decay_lr=0.97, weight_decay=1e-4, model_params=None, loss_fn=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay
        self.model_params = model_params
        self.loss_fn = loss_fn

    def forward(self, x, time):
        y = self.model(x, timespans=time)
        y = y[0] if isinstance(y, tuple) else y

        # 将y按model_params['pred_len']分割
        y = y.reshape(y.shape[0], -1, self.model_params['pred_len'], y.shape[-1])
        return y

    def retrieve_candidate_embed(self, pred_embeds, candidate_embeds) -> [torch.Tensor]:
        """
        Args:
            pred_embeds: [B, T, k, D] tensor of predicted embeddings
            candidate_embeds: [B, T, C, D] tensor of candidate embeddings

        Returns:
            match_embeds: [B, T, k, D] tensor of matched candidate embeddings
        """
        B, T, k, D = pred_embeds.shape
        _, _, C, _ = candidate_embeds.shape

        # Initialize loss func
        mse_loss = self.loss_fn.criterion

        # Reshape tensors for efficient computation
        pred_embeds_flat = pred_embeds.view(B * T * k, D)  # [B*T*k, D]
        candidate_embeds_flat = candidate_embeds.view(B * T * C, D)  # [B*T*C, D]

        # Compute MSE distances between predictions and candidates
        pred_exp = pred_embeds_flat.unsqueeze(1)  # [B*T*k, 1, D]
        cand_exp = candidate_embeds_flat.unsqueeze(0)  # [1, B*T*C, D]
        # Calculate MSE loss for each pair, then mean over D dimension
        mse_distances = mse_loss(pred_exp.expand(-1, B * T * C, -1),
                                 cand_exp.expand(B * T * k, -1, -1))  # [B*T*k, B*T*C, D]
        mse_distances = mse_distances.mean(dim=-1)  # [B*T*k, B*T*C]

        # Reshape distances to separate B and T dimensions
        mse_distances = mse_distances.view(B * T, k, B * T * C)  # [B*T, k, B*T*C]

        # For each B*T, select top k closest candidates
        match_indices = torch.zeros(B * T, k, dtype=torch.long, device=pred_embeds.device)

        for bt in range(B * T):
            # Get the C candidates for this specific B,T pair
            start_idx = (bt // T) * T * C + (bt % T) * C
            end_idx = start_idx + C
            bt_distances = mse_distances[bt, :, start_idx:end_idx]  # [k, C]

            # Get indices of k smallest distances
            _, indices = torch.topk(bt_distances, k, dim=-1, largest=False)
            match_indices[bt] = start_idx + indices

        # Gather the matching embeddings
        match_embeds_flat = candidate_embeds_flat[match_indices]  # [B*T, k, D]
        match_embeds = match_embeds_flat.view(B, T, k, D)  # [B, T, k, D]

        return match_embeds

    def _prepare_batch(self, batch):
        time, event_type_embed, feedback_embed, artifact_embed, candidate_embed, labels_embed, mask = batch

        t_elapsed = time[:, 1:] - time[:, :-1]
        t_fill = torch.zeros(time.size(0), 1, device=time.device)
        time = torch.cat((t_fill, t_elapsed), dim=1)
        time = time * self.model_params['tau']

        x = torch.cat((event_type_embed, feedback_embed, artifact_embed), dim=2)

        return time, x, candidate_embed, labels_embed, mask

    def training_step(self, batch, batch_idx):
        time, x, candidate_embed, labels_embed, mask = self._prepare_batch(batch)

        output = self.forward(x, time)  # 可能返回元组
        artifact_preds = output[0] if isinstance(output, tuple) else output

        # artifact_preds: [B, T, k, D]
        # labels_embed: [B, T, k, D]
        # mask: [B, T, D]
        loss = self.loss_fn(artifact_preds, labels_embed, mask)
        self.log('train_loss', loss)

        artifact_preds = self.retrieve_candidate_embed(artifact_preds, candidate_embed)
        match_num = (artifact_preds == labels_embed).sum(dim=(1, 2))
        match_rate = match_num / self.model_params['pred_len'] / artifact_preds.shape[1]
        self.log('train_acc', match_rate.mean())

        return loss

    def validation_step(self, batch, batch_idx):
        time, x, candidate_embed, labels_embed, mask = self._prepare_batch(batch)

        output = self.forward(x, time)  # 可能返回元组
        artifact_preds = output[0] if isinstance(output, tuple) else output

        loss = self.loss_fn(artifact_preds, labels_embed, mask)
        self.log('val_loss', loss)

        artifact_preds = self.retrieve_candidate_embed(artifact_preds, candidate_embed)
        match_num = (artifact_preds == labels_embed).sum(dim=(1, 2))
        match_rate = match_num / self.model_params['pred_len'] / artifact_preds.shape[1]
        self.log('val_acc', match_rate.mean())

        return loss

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
