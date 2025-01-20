import os
import sys
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from base.log_item import TaskType


# 使用TaskType作为标签
labels = {task_type.value: i for i, task_type in enumerate(TaskType)}

model_params = {
    'in_features': 187,  # 总特征维度: EventType(20) + Artifact(146) + Context(21)
    'out_features': len(TaskType),  # 任务类型数量
    'units': 128,  # 增加隐藏单元数以处理更复杂的特征
    'max_seq_len': 67,  # 保持不变或根据实际序列长度调整
}

train_params = {
    'batch_size': 32,
    'max_epochs': 200,  # 减少轮数，使用早停
    'base_lr': 0.001,  # 降低学习率以提高稳定性
    'decay_lr': 0.97,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'early_stop_patience': 20,  # 减少早停耐心值
    'log_interval': 5
}