import torch

from cfc import CFC
from config import model_params
from artifact_prediction.template.dataset_template import generate_data
from learner import Learner

# 示例问题：根据三角函数的三种组合，生成数据集，其中三角函数的组合包括：
# sin+cos、sin+tan、tan+cos。
# 标签为：0、1、2，分别表示三角函数的组合。

if __name__ == '__main__':
    total_seq_len = 300
    max_seq_len = model_params['max_seq_len']
    data_x, data_y = generate_data(total_seq_len, max_seq_len)
    print(data_x.shape, data_y.shape)

    # 构造模型 
    cfc = CFC(
        in_features=model_params['in_features'],
        out_features=model_params['out_features'],
        units=model_params['units']
    )
    learner = Learner(
        cfc.model,
        lr=0.0,
        decay_lr=0.0,
        weight_decay=0.0,
    )
    learner.load_state_dict(torch.load('ckpt/cfc/version_1/cfc-best.ckpt')['state_dict'])
    learner.eval()

    mean_acc = 0.0
    cnt = len(data_x)

    # 预测
    with torch.no_grad():
        for x, y in zip(data_x, data_y):
            y_hat, _ = learner.model.forward(x)

            enable_signal = torch.sum(y, dim=-1) > 0.0
            y_hat = y_hat[enable_signal]
            y = y[enable_signal]
            y = torch.argmax(y.detach(), dim=-1)

            preds = torch.argmax(y_hat.detach(), dim=-1)
            acc = (preds == y).float().mean()
            print('preds: ', preds)
            print('labels: ', y)
            print(f'acc: {acc:.4f}')
            print('-'*10)
            mean_acc += acc/cnt

    print(f'Mean acc: {mean_acc:.4f}')
