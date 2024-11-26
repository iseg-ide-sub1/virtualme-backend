import random

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from learner import Learner
from cfc import CFC

labels = {
    'sin+cos': 0,
    'sin+tan': 1,
    'tan+cos': 2,
}


def generate_data(seq_len):
    data_x = []
    data_y = []
    # 按照labels生成随机数据
    for label in labels:
        if label == 'sin+cos':
            x = torch.stack([
                torch.sin(torch.linspace(0, 3 * torch.pi, seq_len)),
                torch.cos(torch.linspace(0, 3 * torch.pi, seq_len))
            ]).transpose(0, 1)
        elif label == 'sin+tan':
            x = torch.stack([
                torch.sin(torch.linspace(0, 3 * torch.pi, seq_len)),
                torch.tan(torch.linspace(0, 3 * torch.pi, seq_len))
            ]).transpose(0, 1)
        else:
            x = torch.stack([
                torch.tan(torch.linspace(0, 3 * torch.pi, seq_len)),
                torch.cos(torch.linspace(0, 3 * torch.pi, seq_len))
            ]).transpose(0, 1)
        y = torch.tensor([float(labels[label])] * seq_len)
        data_x.append(x)
        data_y.append(y)

    data_x = torch.cat(data_x)
    data_y = torch.cat(data_y)

    return data_x, data_y


if __name__ == '__main__':
    seq_len = 100
    data_x, data_y = generate_data(seq_len)
    print(data_x.shape, data_y.shape)
    # 构造dataset
    dataset = torch.utils.data.TensorDataset(data_x, data_y)
    # 构造dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True)

    cfc = CFC(in_features=2, out_features=1, units=16)
    learner = Learner(
        cfc.model,
        lr=0.0005,
        decay_lr=0.97,
        weight_decay=1e-4,
    )
    logger = TensorBoardLogger('ckpt', name='cfc', version=1, log_graph=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt/cfc/version_1",  # 指定保存路径
        filename="cfc-{epoch:02d}",  # 文件命名格式
        save_top_k=-1,  # -1表示保存每个epoch的检查点
        save_weights_only=True,  # 仅保存模型权重
        every_n_epochs=100  # 每个epoch保存一次
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        gradient_clip_val=1,
        log_every_n_steps=3,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(learner, dataloader)
