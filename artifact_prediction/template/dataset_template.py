from config import labels
import torch

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