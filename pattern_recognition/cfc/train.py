import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split

from cfc import CFC
from learner import Learner

# 示例问题：根据三角函数的三种组合，生成数据集，其中三角函数的组合包括：
# sin+cos、sin+tan、tan+cos。
# 标签为：0、1、2，分别表示三角函数的组合。
labels = {
    'sin+cos': 0,
    'sin+tan': 1,
    'tan+cos': 2,
}


def generate_data(total_seq_len, max_seq_len):
    data_x = []
    data_y = []
    # 按照labels生成随机数据
    for label in labels:
        if label == 'sin+cos':
            x = torch.stack([
                torch.sin(torch.linspace(0, 3 * torch.pi, total_seq_len)),
                torch.cos(torch.linspace(0, 3 * torch.pi, total_seq_len))
            ]).transpose(0, 1)
        elif label == 'sin+tan':
            x = torch.stack([
                torch.sin(torch.linspace(0, 3 * torch.pi, total_seq_len)),
                torch.tan(torch.linspace(0, 3 * torch.pi, total_seq_len))
            ]).transpose(0, 1)
        else:
            x = torch.stack([
                torch.tan(torch.linspace(0, 3 * torch.pi, total_seq_len)),
                torch.cos(torch.linspace(0, 3 * torch.pi, total_seq_len))
            ]).transpose(0, 1)

        # 标签，使用one-hot编码
        y = torch.zeros(total_seq_len, 3)
        y[:, labels[label]] = 1.0

        data_x.append(x)
        data_y.append(y)

    all_x = []
    all_y = []

    # 按照max_seq_len切分数据为多个块
    for x, y in zip(data_x, data_y):
        for i in range(0, total_seq_len, max_seq_len):
            end = min(i + max_seq_len, total_seq_len)
            if end - i < max_seq_len:
                continue
            all_x.append(x[i:end])
            all_y.append(y[i:end])

    # 转换为Tensor
    dataset_x = torch.stack(all_x)  # 形状: (样本数, max_seq_len, 2)
    dataset_y = torch.stack(all_y)  # 形状: (样本数, max_seq_len, 3)

    return dataset_x, dataset_y


if __name__ == '__main__':
    total_seq_len = 2000
    max_seq_len = 67  # 不能整除，打乱使得每个样本包含不同label
    data_x, data_y = generate_data(total_seq_len, max_seq_len)
    print(data_x.shape, data_y.shape)
    # 构造dataset
    dataset = torch.utils.data.TensorDataset(data_x, data_y)
    # 划分数据集
    train_size = int(0.8 * len(dataset))  # 80% 训练数据
    val_size = len(dataset) - train_size  # 20% 验证数据
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 构造训练和验证 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    cfc = CFC(in_features=2, out_features=3, units=32)
    learner = Learner(
        cfc.model,
        lr=0.02,
        decay_lr=0.97,
        weight_decay=1e-4,
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
        patience=50,  # 如果验证准确度n个epoch没有提升，则停止训练
        verbose=True
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        gradient_clip_val=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(
        learner,
        train_dataloaders=train_dataloader,
         val_dataloaders=val_dataloader
    )
