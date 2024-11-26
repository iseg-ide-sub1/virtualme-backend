import pytorch_lightning as pl
import torch
import torch.nn as nn


class Learner(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, decay_lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_lr = decay_lr
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)

        enable_signal = torch.sum(y, -1) > 0.0
        y_hat = y_hat[enable_signal]
        y = y[enable_signal]
        y = torch.argmax(y.detach(), dim=-1)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        preds = torch.argmax(y_hat.detach(), dim=-1)  # labels are given as one-hot
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)

        enable_signal = torch.sum(y, -1) > 0.0
        y_hat = y_hat[enable_signal]
        y = y[enable_signal]
        y = torch.argmax(y.detach(), dim=-1)
        loss = nn.CrossEntropyLoss()(y_hat, y)

        preds = torch.argmax(y_hat.detach(), dim=-1)  # labels are given as one-hot
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.validation_step_outputs.append((loss, acc))
        return loss, acc

    def on_validation_epoch_end(self):
        val_acc = torch.stack([l[1] for l in self.validation_step_outputs])

        val_acc = torch.mean(val_acc)
        print(f"\nval_acc: {val_acc.item():0.3f}\n")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: self.decay_lr ** epoch
        )
        return [optimizer], [scheduler]
