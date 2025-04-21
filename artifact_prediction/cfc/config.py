

model_params = {
    'event_type_embedding_dim': 50,  # 事件类型嵌入维度
    'feedback_dim': 1,  # 反馈维度
    'artifact_embedding_dim': 27,  # 工件特征嵌入维度
    'units': 128,  # 隐藏单元数
    'max_seq_len': 50,  # 根据实际序列长度调整
    'candidate_num': 10,  # 候选集大小
    'pred_len': 1,  # 预测步长
    'tau': 1,  # 时间戳缩放倍率
}

train_params = {
    'batch_size': 1,
    'max_epochs': 200,
    'base_lr': 0.001,
    'decay_lr': 0.97,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'train_ratio': 0.8,
    'val_ratio': 0.2,
    'early_stop_patience': 20,
    'log_interval': 5
}