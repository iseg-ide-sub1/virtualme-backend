try:
    from .event_types import EventType
    from .config import wv_params
except ImportError:
    from event_types import EventType
    from config import wv_params

import os
import re
from typing import List

import torch
from gensim.models import Word2Vec


def build_word2vec_model() -> Word2Vec:
    # 提取所有EventType短语
    event_phrases = [event.value for event in EventType]

    def split_words(phrase: str) -> List[str]:
        """
        对短语进行分词处理，支持驼峰命名和下划线分隔

        Args:
            phrase: 输入短语（如"OpenTextDocument"或"filesExplorer_copy"）

        Returns:
            List[str]: 分词后的单词列表（小写）
        """
        # 先按下划线分割
        phrase = phrase[0] if isinstance(phrase, tuple) else phrase
        parts = phrase.split('_')

        # 对每个部分处理驼峰命名
        words = []
        for part in parts:
            if part:  # 跳过空字符串
                # 使用正则表达式拆分驼峰命名
                camel_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part)
                words.extend(camel_words)

        # 转换为小写并过滤空结果
        return [word.lower() for word in words if word]

    # 创建训练语料库
    corpus = [split_words(phrase) for phrase in event_phrases]

    # 训练Word2Vec模型
    model = Word2Vec(
        sentences=corpus,
        vector_size=wv_params["vector_size"],  # 词向量维度
        window=wv_params["window"],  # 上下文窗口
        min_count=wv_params["min_count"],  # 最小词频
        workers=wv_params["workers"],  # 并行训练线程
        sg=wv_params["sg"],  # 使用skip-gram
        hs=wv_params["hs"],  # 使用negative sampling
        negative=wv_params["negative"],  # 负采样数量
        epochs=wv_params["epochs"]  # 迭代次数
    )

    # 创建保存目录并保存模型
    model_dir = os.path.join(os.path.dirname(__file__), "wv")
    model_path = os.path.join(model_dir, "event_type_word2vec.model")

    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)

    return model


def event_type_2_vec(event_type: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    model_path = os.path.join(os.path.dirname(__file__), "wv", "event_type_word2vec.model")
    if not os.path.exists(model_path):
        print("Word2Vec model found. Building...")
        build_word2vec_model()

    # 加载训练好的模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Word2Vec model not found at {model_path}. Please run build_word2vec_model() first.")

    model = Word2Vec.load(model_path)

    # 分词处理
    def split_camel_case(phrase: str) -> List[str]:
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', phrase)
        return [word.lower() for word in words]

    # 获取短语的词列表
    words = split_camel_case(event_type)

    # 计算短语的平均嵌入
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(torch.tensor(model.wv[word], device=device))

    if not vectors:
        # 如果没有词在模型中，返回零向量
        return torch.zeros(model.vector_size)

    # 转换为torch.Tensor并取平均
    embedding = torch.stack(vectors).mean(dim=0)

    return embedding


# 示例用法
if __name__ == "__main__":

    # 遍历所有事件类型，打印
    print(len(EventType))
    print("All event types:")
    for event_type in EventType:
        # 打印事件类型名称
        print(event_type.value)

    # 测试转换
    sample_event = "OpenTextDocument"
    embedding = event_type_2_vec(sample_event)
    print(f"Event: {sample_event}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}...")
