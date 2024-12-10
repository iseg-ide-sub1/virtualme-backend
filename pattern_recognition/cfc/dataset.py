# pattern_recognition/cfc/dataset.py

import torch
import pickle
import numpy as np
from typing import Tuple, List
from gensim.models import Word2Vec
import os
import sys
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from base.log_item import Log, LogItem, EventType, TaskType, ArtifactType, ContextType

def load_log(pt_file: str) -> Log:
    """加载预处理后的.pt文件"""
    with open(pt_file, 'rb') as f:
        return pickle.load(f)

class ArtifactEmbedding:
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.model = None
        
    def train_embeddings(self, log: Log):
        """从日志数据训练词嵌入模型"""
        sentences = []
        for item in log.log_items:
            if item.artifact:
                words = self._process_name(item.artifact.name)
                if words:
                    sentences.append(words)
        
        self.model = Word2Vec(sentences=sentences, 
                            vector_size=self.vector_size,
                            window=5,
                            min_count=1,
                            workers=4)
    
    def _process_name(self, name: str) -> List[str]:
        """处理名称，分割为词序列"""
        words = []
        # 处理路径分隔符
        for part in name.split('->'):
            # 处理驼峰命名
            current_word = ""
            for char in part:
                if char.isupper() and current_word:
                    words.append(current_word.lower())
                    current_word = char
                else:
                    current_word += char
            if current_word:
                words.append(current_word.lower())
            
            # 处理下划线命名
            words.extend(part.split('_'))
        
        return [w for w in words if w]  # 移除空字符串
    
    def get_embedding(self, name: str) -> List[float]:
        """获取名称的嵌入向量"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        words = self._process_name(name)
        vectors = []
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0).tolist()
        return [0.0] * self.vector_size

def encode_event_type(event_type: EventType) -> List[float]:
    """将EventType编码为one-hot向量"""
    encoding = [0.0] * len(EventType)
    encoding[list(EventType).index(event_type)] = 1.0
    return encoding

def encode_artifact(artifact, embedding_model: ArtifactEmbedding) -> List[float]:
    """将Artifact编码为特征向量"""
    if artifact is None:
        # 增加特征维度：基础特征 + 语义特征 + 嵌入特征
        return [0.0] * (len(ArtifactType) + 15 + embedding_model.vector_size)
    
    # 1. 制品类型的one-hot编码
    type_encoding = [0.0] * len(ArtifactType)
    type_encoding[list(ArtifactType).index(artifact.artifact_type)] = 1.0
    
    # 2. 基础名称特征
    name_parts = artifact.name.split('->')
    current_node = name_parts[-1]
    
    name_features = [
        float(len(artifact.name)),      # 名称总长度
        float(len(name_parts)),         # 层级深度
        float(len(current_node))        # 当前节点名称长度
    ]
    
    # 3. 路径特征
    path_features = [
        1.0 if any(part.endswith('.py') for part in name_parts) else 0.0,    # Python文件
        1.0 if any(part.endswith('.json') for part in name_parts) else 0.0,  # JSON文件
        1.0 if any(part.endswith('.txt') for part in name_parts) else 0.0,   # 文本文件
        1.0 if any(part.endswith('.md') for part in name_parts) else 0.0,    # Markdown文件
        1.0 if any(part.startswith('test') for part in name_parts) else 0.0  # 测试文件
    ]
    
    # 4. 命名规范特征
    naming_features = [
        1.0 if current_node.islower() else 0.0,                    # 全小写
        1.0 if current_node.isupper() else 0.0,                    # 全大写
        1.0 if '_' in current_node else 0.0,                       # 下划线命名
        1.0 if any(c.isupper() for c in current_node[1:]) else 0.0 # 驼峰命名
    ]
    
    # 5. 引用特征
    reference_features = [
        1.0 if artifact.reference else 0.0,  # 是否有引用
        float(len(artifact.reference)) if artifact.reference else 0.0,  # 引用数量
    ]
    
    # 6. Word2Vec嵌入特征 ?
    embedding_features = embedding_model.get_embedding(artifact.name)
    
    # 组合所有特征
    return (type_encoding + name_features + path_features + 
            naming_features + reference_features + embedding_features)

def encode_context(context) -> List[float]:
    """将Context编码为特征向量，增加语义特征"""
    if context is None:
        return [0.0] * (len(ContextType) + 12)  # 增加特征维度
    
    # 1. 上下文类型的one-hot编码
    type_encoding = [0.0] * len(ContextType)
    type_encoding[list(ContextType).index(context.context_type)] = 1.0
    
    # 2. 内容长度特征
    content_before, content_after = context.content
    length_features = [
        float(len(content_before)),  # before内容长度
        float(len(content_after)),   # after内容长度
        float(abs(len(content_after) - len(content_before)))  # 内容变化量
    ]
    
    # 3. 位置特征
    start_line, start_char = context.start
    end_line, end_char = context.end
    position_features = [
        float(start_line),
        float(start_char),
        float(end_line),
        float(end_char),
        float(end_line - start_line + 1)  # 跨越的行数
    ]
    
    # 4. 终端命令特征（如果是终端相关）
    terminal_features = []
    if context.context_type == ContextType.Terminal:
        cmd = content_after.strip()
        terminal_features = [
            1.0 if cmd.startswith('git ') else 0.0,     # 是否git命令
            1.0 if cmd.startswith('python ') else 0.0,  # 是否python命令
            1.0 if '|' in cmd else 0.0,                 # 是否管道命令
            1.0 if '>' in cmd else 0.0                  # 是否重定向
        ]
    else:
        terminal_features = [0.0] * 4
    
    return (type_encoding + length_features + position_features + terminal_features)
    """将Context编码为特征向量"""
    if context is None:
        return [0.0] * (len(ContextType) + 2)  # context_type + 2个长度特征
    
    # 上下文类型的one-hot编码
    type_encoding = [0.0] * len(ContextType)
    type_encoding[list(ContextType).index(context.context_type)] = 1.0
    
    # 内容长度特征
    content_features = [
        float(len(context.content[0])),  # before内容长度
        float(len(context.content[1]))   # after内容长度
    ]
    
    return type_encoding + content_features

def encode_task_type(task_type: TaskType) -> int:
    """将TaskType编码为类别索引"""
    return list(TaskType).index(task_type)

def generate_data(dataset_dir: str, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成训练数据张量，从目录中读取所有pt文件"""
    all_sequences_x = []
    all_sequences_y = []
    
    # 获取目录下所有的pt文件
    pt_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]
    print(f"Found {len(pt_files)} pt files in {dataset_dir}")
    
    for pt_file in pt_files:
        pt_path = os.path.join(dataset_dir, pt_file)
        print(f"Processing {pt_file}...")
        
        # 读取单个文件的数据
        log = load_log(pt_path)
        
        # 初始化并训练Word2Vec模型
        embedding_model = ArtifactEmbedding(vector_size=100)
        embedding_model.train_embeddings(log)
        
        current_sequence_x = []
        current_sequence_y = []
        
        for item in log.log_items:
            features = (
                encode_event_type(item.event_type) +
                encode_artifact(item.artifact, embedding_model) +
                encode_context(item.context)
            )
            
            label = encode_task_type(item.task_type)
            
            current_sequence_x.append(features)
            current_sequence_y.append(label)
            
            if len(current_sequence_x) == max_seq_len:
                all_sequences_x.append(current_sequence_x)
                all_sequences_y.append(current_sequence_y)
                current_sequence_x = []
                current_sequence_y = []
        
        # 处理最后一个不完整序列（如果存在且长度足够）
        if len(current_sequence_x) >= max_seq_len // 2:
            # 填充到最大长度
            feature_dim = len(all_sequences_x[0][0]) if all_sequences_x else len(current_sequence_x[0])
            padding_length = max_seq_len - len(current_sequence_x)
            current_sequence_x.extend([[0.0] * feature_dim] * padding_length)
            current_sequence_y.extend([0] * padding_length)
            all_sequences_x.append(current_sequence_x)
            all_sequences_y.append(current_sequence_y)
    
    if not all_sequences_x:
        raise ValueError("No sequences generated from the dataset")
    
    # 转换为张量
    dataset_x = torch.tensor(all_sequences_x, dtype=torch.float32)
    dataset_y = torch.tensor(all_sequences_y, dtype=torch.long)
    
    # 对特征进行归一化
    mean = dataset_x.mean(dim=(0, 1), keepdim=True)
    std = dataset_x.std(dim=(0, 1), keepdim=True)
    dataset_x = (dataset_x - mean) / (std + 1e-8)
    
    return dataset_x, dataset_y
def get_task_type_count() -> int:
    """获取TaskType的总数"""
    return len(TaskType)

if __name__ == '__main__':
    pt_file = 'dataset/2024-12-09 19.pt'
    max_seq_len = 67  # 与模型配置保持一致
    
    dataset_x, dataset_y = generate_data(pt_file, max_seq_len)
    print(f'Dataset X shape: {dataset_x.shape}')  # [样本数, 序列长度, 特征数]
    print(f'Dataset Y shape: {dataset_y.shape}')  # [样本数, 序列长度]
    
    # 打印一些统计信息
    print(f'Number of sequences: {len(dataset_x)}')
    print("Feature dimensions:")
    print(f"- EventType: {len(EventType)}")  # 20
    print(f"- Artifact: {len(ArtifactType) + 15 + 100}")  # 146
    print(f"- Context: {len(ContextType) + 12}")  # 21
    print(f"Total: {len(EventType) + (len(ArtifactType) + 15 + 100) + (len(ContextType) + 12)}")  # 187
    print(f'Number of task types: {get_task_type_count()}')