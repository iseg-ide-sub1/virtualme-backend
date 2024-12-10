# pattern_recognition/cfc/learner.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class Learner(pl.LightningModule):
    def __init__(self, model, lr=0.001, decay_lr=0.97, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)  # 可能返回元组
        y_hat = output[0] if isinstance(output, tuple) else output
        
        # 重塑张量以适应CrossEntropyLoss
        y_hat = y_hat.view(-1, y_hat.size(-1))  # [batch_size * seq_len, num_classes]
        y = y.view(-1)  # [batch_size * seq_len]
        
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        
        # 计算准确率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)  # 可能返回元组
        y_hat = output[0] if isinstance(output, tuple) else output
        
        # 重塑张量以适应CrossEntropyLoss
        y_hat = y_hat.view(-1, y_hat.size(-1))  # [batch_size * seq_len, num_classes]
        y = y.view(-1)  # [batch_size * seq_len]
        
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss)
        
        # 计算准确率
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc)
        
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