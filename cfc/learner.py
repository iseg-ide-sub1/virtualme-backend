import pytorch_lightning as pl
import torch
import torch.nn as nn


class Learner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)

        loss = nn.L1Loss()(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
