import json
import os
from datetime import datetime

import torch

try:
    from .config import *
    from .event_type_2_vec import event_type_2_vec
except ImportError:
    from config import *
    from event_type_2_vec import event_type_2_vec


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, model_params):
        self.model_params = model_params
        self.seqs = []

    def load_train_data_from_raw(self, train_data_folder):
        # 读取train_data_folder下的所有json文件名list
        train_json_list = [file_name for file_name in os.listdir(train_data_folder) if file_name.endswith('.json')]

        # 文件命名模式：v0.3.0_2025-04-20-16.05.54.029_train.json
        # 其中v0.3.0为版本号，2025-04-20-16.05.54.029为时间戳，train为数据类型，按照时间戳从早到晚排序
        train_json_list.sort(key=lambda x: x.split('_')[1])

        # 读取json文件内容，并将其转换为list
        train_data_list = []
        for file_name in train_json_list:
            with open(os.path.join(train_data_folder, file_name), 'r', encoding='utf-8') as f:
                train_data_list.extend(json.load(f))

        pred_len = self.model_params['pred_len']
        max_seq_len = self.model_params['max_seq_len']

        # 从raw构造dataset
        times = []
        event_types = []
        feedbacks = []
        artifact_embeds = []
        candidate_embeds_list = []
        labels_list = []

        first_time = None

        for i, event in enumerate(train_data_list):
            if i == train_data_list.__len__() - pred_len:  # 最后pred_len个event没有下pred_len个event作为label，因此不用处理
                break

            # 时间戳========================================================================================================
            timestamp = event['timestamp']
            # 使用Date库将timestamp转换为秒数，timestamp是iso格式的字符串，如'2021-08-17T10:30:00.000Z'
            timestamp = datetime.fromisoformat(timestamp).timestamp()
            if first_time is None:
                first_time = timestamp
            # 计算序列的相对时间,单位为300毫秒
            time = torch.tensor(float((timestamp - first_time) * 1000 / 300))

            # 事件类型======================================================================================================
            event_type = event['eventType']
            event_type_embed = event_type_2_vec(event_type)

            # 环境反馈======================================================================================================
            feedback = torch.tensor(event['feedback']).unsqueeze(0)

            # 工件嵌入======================================================================================================
            if event['artifactEmbed'] is not None:
                artifact_embed = torch.tensor(event['artifactEmbed'])
            else:
                artifact_embed = torch.zeros(model_params['artifact_embedding_dim'])

            # 候选工件======================================================================================================
            if event['candidateEmbeds'] is None:
                candidate_embeds = [torch.zeros_like(artifact_embed)] * model_params['candidate_num']
            else:
                candidate_embeds = [torch.tensor(embed) for embed in event['candidateEmbeds']]
            # 保证candidate_embeds的长度为model_params['candidate_num']
            if len(candidate_embeds) < model_params['candidate_num']:
                candidate_embeds.extend(
                    [torch.zeros_like(artifact_embed)] * (model_params['candidate_num'] - len(candidate_embeds)))
            else:
                candidate_embeds = candidate_embeds[:model_params['candidate_num']]

            # 标签======================================================================================================
            # 以下计算保证了label来源于当前时间步的候选工件嵌入，在计算acc时，模型输出与候选匹配检索后直接与标签比较
            def artifact_equal(a, b) -> bool:
                if a is None or b is None:
                    return False
                if a['name'] is None or b['name'] is None:
                    return False
                if a['startPosition'] is None or b['startPosition'] is None:
                    return False
                if a['endPosition'] is None or b['endPosition'] is None:
                    return False
                return a['name'] == b['name'] and a['startPosition'] == b['startPosition'] and a['endPosition'] == b[
                    'endPosition']

            labels: list[torch.Tensor] = []
            for j in range(pred_len):
                next_event = train_data_list[i + j + 1]
                if next_event['artifact'] is None:
                    labels.append(torch.zeros(model_params['artifact_embedding_dim']))
                    continue
                candidate_artifacts = event['candidates']
                if candidate_artifacts is None:
                    labels.append(torch.zeros(model_params['artifact_embedding_dim']))
                    continue
                # 找到候选工件中与下一个事件工件相同的索引,将其对应的候选工件嵌入作为标签，如果没有找到，则用0填充
                found = False
                for k, candidate_artifact in enumerate(candidate_artifacts):
                    if artifact_equal(candidate_artifact, next_event['artifact']):
                        labels.append(candidate_embeds[k])
                        found = True
                        break
                if not found:
                    labels.append(torch.zeros(model_params['artifact_embedding_dim']))


            # 将数据添加到各自列表中
            times.append(time)
            feedbacks.append(feedback)
            event_types.append(event_type_embed)
            artifact_embeds.append(artifact_embed)
            candidate_embeds_list.append(candidate_embeds)
            labels_list.append(labels)

        assert len(times) == len(event_types) == len(artifact_embeds) == len(candidate_embeds_list) == len(
            labels_list) == len(feedbacks)

        seqs = []
        times = torch.stack(times)
        feedbacks = torch.stack(feedbacks)
        event_types = torch.stack(event_types)
        artifact_embeds = torch.stack(artifact_embeds)
        candidate_embeds_list = torch.stack([torch.stack(embeds) for embeds in candidate_embeds_list])
        labels_list = torch.stack([torch.stack(labels) for labels in labels_list])

        print('times:', times.size())
        print('event_types:', event_types.size())
        print('feedbacks:', feedbacks.size())
        print('artifact_embeds:', artifact_embeds.size())
        print('candidate_embeds_list:', candidate_embeds_list.size())
        print('labels_list:', labels_list.size())

        # 按max_seq_len将数据分割为多个序列，每个seq是6个序列的元组，包含时间戳、事件类型、反馈、工件嵌入、候选工件嵌入、标签
        seq_length = len(times)
        offset = 0
        slide = max_seq_len // 2
        while offset + max_seq_len < seq_length:
            idx = range(offset, offset + max_seq_len)

            first_tp = times[idx][0]
            seqs.append(
                (
                    times[idx] - first_tp, event_types[idx], feedbacks[idx], artifact_embeds[idx],
                    candidate_embeds_list[idx],
                    labels_list[idx])
            )
            offset += slide

        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

    def __repr__(self):
        ret = f'EventDataset({len(self.seqs)} sequences)\n'
        ret += f'Each sequence has max length {self.seqs[0][0].size(0)} events\n'
        ret += f'Each event has {len(self.seqs[0])} features, including time, event_type, feedback, artifact_embed, candidate_embeds, and labels\n'
        times, event_types, feedbacks, artifact_embeds, candidate_embeds_list, labels_list = self.seqs[0]
        ret += f'For example, an event has the following features:\n'
        ret += f'Time: 1 dim tensor\n'
        ret += f'Event type: {event_types.size(1)} dim tensor\n'
        ret += f'Feedback: 1 dim tensor\n'
        ret += f'Artifact embed: {artifact_embeds.size(1)} dim tensor\n'
        ret += f'In-feature total: {event_types.size(1) + 1 + artifact_embeds.size(1)} dim tensor\n'
        ret += f'Candidate embeds: {candidate_embeds_list.size(1)} 个 {candidate_embeds_list.size(2)} dim tensor\n'
        ret += f'Labels: {labels_list.size(1)} 个 {labels_list.size(2)} dim tensor\n'
        ret += f'Model params: {self.model_params}\n'

        return ret


if __name__ == '__main__':
    event_dataset = EventDataset(model_params)
    event_dataset.load_train_data_from_raw("../../dataset_raw")
    print(event_dataset)
