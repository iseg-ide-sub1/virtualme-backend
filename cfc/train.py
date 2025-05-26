import datetime
import os
import shutil

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# 兼容从自身目录运行
try:
    from .dataset import EventDataset
    from .config import train_params, model_params
    from .CfC import CFC
except ImportError:
    from dataset import EventDataset
    from config import train_params, model_params
    from CfC import CFC

set_float32_matmul_precision('medium')


def train(dataset_dir='./dataset_raw', date_filter: datetime.datetime = None):
    batch_size = train_params['batch_size']
    num_workers = train_params['num_workers']

    # 加载数据, 构造dataset
    event_dataset = EventDataset()
    event_dataset.load_train_data_from_raw(dataset_dir, date_filter)
    print(event_dataset)
    train_size = int(train_params['train_ratio'] * len(event_dataset))
    val_size = len(event_dataset) - train_size
    train_dataset, val_dataset = random_split(event_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=True
    )

    cfc = CFC(
        lr=train_params['base_lr'],
        decay_lr=train_params['decay_lr'],
        weight_decay=train_params['weight_decay'],
    )

    # 配置日志记录器
    logger = TensorBoardLogger('tensorboard', name='cfc')

    # 配置检查点和早停回调
    # 获取当前日期
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt",
        filename=now,
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_weights_only=False
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=train_params['early_stop_patience'],
        verbose=True
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=train_params['max_epochs'],
        gradient_clip_val=1,
        log_every_n_steps=train_params['log_interval'],
        callbacks=[checkpoint_callback, early_stop_callback],
        num_sanity_val_steps=0
    )

    # 开始训练
    trainer.fit(
        cfc,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 打印训练结果
    print(f'Best validation accuracy: {checkpoint_callback.best_model_score:.4f}')

if __name__ == '__main__':
    train()
