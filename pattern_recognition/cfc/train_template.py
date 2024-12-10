import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split

from cfc import CFC
from config_template import train_params, model_params
from learner import Learner
from dataset_template import generate_data


# 示例问题：根据三角函数的三种组合，生成数据集，其中三角函数的组合包括：
# sin+cos、sin+tan、tan+cos。
# 标签为：0、1、2，分别表示三角函数的组合。


if __name__ == '__main__':
    total_seq_len = 2000
    max_seq_len = model_params['max_seq_len']
    batch_size = train_params['batch_size']
    num_workers = train_params['num_workers']
    data_x, data_y = generate_data(total_seq_len, max_seq_len)
    print(data_x.shape, data_y.shape)

    # 构造dataset
    dataset = torch.utils.data.TensorDataset(data_x, data_y)
    train_size = int(train_params['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    cfc = CFC(
        in_features=model_params['in_features'],
        out_features=model_params['out_features'],
        units=model_params['units']
    )

    learner = Learner(
        cfc.model,
        lr=train_params['base_lr'],
        decay_lr=train_params['decay_lr'],
        weight_decay=train_params['weight_decay'],
    )
    logger = TensorBoardLogger('ckpt', name='cfc', version=1, log_graph=True)

    # 配置检查点和早停回调
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt/cfc/version_1",
        filename="cfc-best",  # 保存最佳模型
        save_top_k=1,  # 仅保存性能最佳的检查点
        monitor="val_acc",  # 监控验证准确度
        mode="max",  # 验证准确度最大化
        save_weights_only=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",  # 监控验证准确度
        mode="max",  # 希望验证准确度最大化
        patience=train_params['early_stop_patience'],  # 如果验证准确度n个epoch没有提升，则停止训练
        verbose=True
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=train_params['max_epochs'],
        gradient_clip_val=1,
        log_every_n_steps=train_params['log_interval'],
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(
        learner,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
