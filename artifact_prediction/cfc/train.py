import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split

# 兼容从自身目录运行
try:
    from .dataset import EventDataset
    from .cfc import CFC
    from .config import train_params, model_params
    from .learner import Learner, MultiStepLoss
except ImportError:
    from dataset import EventDataset
    from cfc import CFC
    from config import train_params, model_params
    from learner import Learner, SequenceMultiStepLoss

torch.set_float32_matmul_precision('medium')


if __name__ == '__main__':
    dataset_dir = '../../dataset_raw'
    max_seq_len = model_params['max_seq_len']
    pred_len = model_params['pred_len']
    batch_size = train_params['batch_size']
    num_workers = train_params['num_workers']

    # 加载数据, 构造dataset
    event_dataset = EventDataset()
    event_dataset.load_train_data_from_raw(dataset_dir)
    train_size = int(train_params['train_ratio'] * len(event_dataset))
    val_size = len(event_dataset) - train_size
    train_dataset, val_dataset = random_split(event_dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )

    # 构造模型
    cfc = CFC()

    learner = Learner(
        cfc.model,
        lr=train_params['base_lr'],
        decay_lr=train_params['decay_lr'],
        weight_decay=train_params['weight_decay'],
        model_params=model_params,
        loss_fn=SequenceMultiStepLoss()
    )

    # 配置日志记录器
    logger = TensorBoardLogger('ckpt', name='cfc', version=1, log_graph=True)

    # 配置检查点和早停回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt/cfc/version_1",
        filename="cfc-best",
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        save_weights_only=True
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
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # 开始训练
    trainer.fit(
        learner,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 打印训练结果
    print(f'Best validation accuracy: {checkpoint_callback.best_model_score:.4f}')
