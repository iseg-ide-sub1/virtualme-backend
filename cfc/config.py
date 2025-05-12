

model_params = {
    'event_type_embedding_dim': 16,  # 事件类型嵌入维度
    'feedback_dim': 1,  # 反馈维度
    'artifact_embedding_dim': 18,  # 工件特征嵌入维度
    'units': 128,  # 隐藏单元数
    'max_seq_len': 5,  # 根据实际序列长度调整
    'candidate_num': 10,  # 候选集大小
    'tau': 1,  # 时间戳缩放倍率
    'k': 5, # rank排名前k作为模型推理输出
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
    'log_interval': 2
}

wv_params = {
    'vector_size': 16,  # 词向量维度
    'window': 5,       # 上下文窗口
    'min_count': 1,    # 最小词频
    'workers': 4,      # 并行训练线程
    'sg': 1,          # 使用skip-gram
    'hs': 0,          # 使用negative sampling
    'negative': 5,     # 负采样数量
    'epochs': 30      # 迭代次数
}