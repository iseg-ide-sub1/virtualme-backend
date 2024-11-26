import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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

    # 创建最终的数据集
    all_x = []
    all_y = []

    # 按照max_seq_len切分数据为多个块
    for x, y in zip(data_x, data_y):
        for i in range(0, total_seq_len, max_seq_len):
            end = min(i + max_seq_len, total_seq_len)
            all_x.append(x[i:end])
            all_y.append(y[i:end])

    # 转换为Tensor
    dataset_x = torch.stack(all_x)  # 形状: (样本数, max_seq_len, 2)
    dataset_y = torch.stack(all_y)  # 形状: (样本数, max_seq_len, 3)

    return dataset_x, dataset_y


if __name__ == '__main__':
    total_seq_len = 2000
    max_seq_len = 50
    data_x, data_y = generate_data(total_seq_len, max_seq_len)
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

    cfc = CFC(in_features=2, out_features=3, units=32)
    learner = Learner(
        cfc.model,
        lr=0.02,
        decay_lr=0.97,
        weight_decay=1e-4,
    )
    logger = TensorBoardLogger('ckpt', name='cfc', version=1, log_graph=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt/cfc/version_1",  # 指定保存路径
        filename="cfc-{epoch:02d}",  # 文件命名格式
        save_top_k=-1,  # -1表示保存每个epoch的检查点
        save_weights_only=True,  # 仅保存模型权重
        every_n_epochs=100  # 每n个epoch保存一次
    )
    trainer = Trainer(
        logger=logger,
        max_epochs=400,
        gradient_clip_val=1,
        log_every_n_steps=3,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(learner, dataloader)
