import json
import os
import random
from datetime import datetime
from typing import List

import torch.nn.functional as F
import torch

try:
    from .config import *
    from .event_type_2_vec import event_type_2_vec
    from .config import model_params
except ImportError:
    from config import *
    from event_type_2_vec import event_type_2_vec
    from config import model_params

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def encode_an_event(first_time, event, next_event=None, is_inference=False):
    # 时间戳========================================================================================================
    timestamp = event['timestamp']
    # 使用Date库将timestamp转换为秒数，timestamp是iso格式的字符串，如'2021-08-17T10:30:00.000Z'
    timestamp = datetime.fromisoformat(timestamp).timestamp()
    # 计算序列的相对时间,单位为300毫秒
    time = torch.tensor(float((timestamp - first_time) * 1000 / 300), device=device)

    # 事件类型======================================================================================================
    event_type = event['eventType']
    event_type_embed = event_type_2_vec(event_type, device=device)

    # 环境反馈======================================================================================================
    feedback = torch.tensor(event['feedback'], device=device).unsqueeze(0)

    # 工件嵌入======================================================================================================
    if event['artifactEmbed'] is not None:
        artifact_embed = torch.tensor(event['artifactEmbed'], device=device)
    else:
        artifact_embed = torch.zeros(model_params['artifact_embedding_dim'], device=device)

    # 候选工件的嵌入======================================================================================================
    if event['candidateEmbeds'] is None:
        candidate_embeds = ([torch.zeros_like(artifact_embed, device=device)] * model_params['candidate_num'])
    else:
        candidate_embeds = [torch.tensor(embed, device=device, dtype=torch.float) for embed in event['candidateEmbeds']]
        # 对每个嵌入向量进行归一化
        candidate_embeds = [F.normalize(embed, p=2, dim=-1) for embed in candidate_embeds]

    # 保证candidate_embeds的长度为model_params['candidate_num']
    if len(candidate_embeds) < model_params['candidate_num']:
        candidate_embeds.extend(
            [torch.ones_like(artifact_embed, device=device) * 10.0] * (model_params['candidate_num'] - len(candidate_embeds)))
        # 填充候选工件的嵌入手动设置为10， 保证模型不会输出这些值
    else:
        candidate_embeds = candidate_embeds[:model_params['candidate_num']]

    if is_inference:
        return (time,
                event_type_embed,
                feedback,
                artifact_embed,
                candidate_embeds,
                torch.zeros_like(artifact_embed, device=device),
                torch.tensor(1, device=device).unsqueeze(0))

    # 标签======================================================================================================
    # 以下计算保证了label来源于当前时间步的候选工件嵌入，在计算acc时，模型输出与候选匹配检索后，再与标签比较
    def artifact_equal(a, b) -> bool:
        if a is None or b is None:
            return False
        if a['name'] is None or b['name'] is None:
            return False
        if a['type'] is None or b['type'] is None:
            return False
        return True

    label = torch.zeros(model_params['artifact_embedding_dim'], device=device)
    if next_event['artifact'] is None or event['candidates'] is None:
        label = torch.zeros(model_params['artifact_embedding_dim'], device=device)
    else:
        candidate_artifacts = [candidate for candidate in event['candidates']]
        # 找到候选工件中与下一个事件工件相同的索引,将其对应的候选工件嵌入作为标签，如果没有找到，则用0填充
        for k, candidate_artifact in enumerate(candidate_artifacts):
            if artifact_equal(candidate_artifact, next_event['artifact']):
                label = candidate_embeds[k]
                break

    # 掩码======================================================================================================
    # mask默认为0即label无效，连续工件的mask为1，跨工件的mask为2
    mask: torch.Tensor
    if label.sum() == 0:  # 如果label全0，说明该步理论上无法预测
        mask = torch.tensor(0, device=device).unsqueeze(0)
    elif next_event is not None and not artifact_equal(event['artifact'], next_event['artifact']):
        mask = torch.tensor(2, device=device).unsqueeze(0)
    else:
        mask = torch.tensor(1, device=device).unsqueeze(0)

    return time, event_type_embed, feedback, artifact_embed, candidate_embeds, label, mask


def encode_from_json(events, is_inference=False) -> List[tuple] | tuple:
    # 从raw构造dataset
    if events.__len__() > model_params['max_seq_len'] and is_inference:
        events = events[-model_params['max_seq_len']:]
        print('[Inference mode]events truncated to max_seq_len:', model_params['max_seq_len'])

    times = []
    event_types = []
    feedbacks = []
    artifact_embeds = []
    candidate_embeds_list = []
    labels = []
    masks = []

    first_time = datetime.fromisoformat(events[0]['timestamp']).timestamp()

    for i, event in enumerate(events):
        if i == events.__len__() - 1 and not is_inference:  # 最后1个event没有下1个event作为label，因此不用处理
            break

        next_event = None if is_inference else events[i + 1]

        time, event_type_embed, feedback, artifact_embed, candidate_embeds, label, mask = (
            encode_an_event(first_time, event, next_event, is_inference))

        # 将数据添加到各自列表中
        times.append(time)
        feedbacks.append(feedback)
        event_types.append(event_type_embed)
        artifact_embeds.append(artifact_embed)
        candidate_embeds_list.append(candidate_embeds)
        labels.append(label)
        masks.append(mask)

    assert len(times) == len(event_types) == len(artifact_embeds) == len(candidate_embeds_list) == len(
        labels) == len(feedbacks) == len(masks)

    times = torch.stack(times)
    feedbacks = torch.stack(feedbacks)
    event_types = torch.stack(event_types)
    artifact_embeds = torch.stack(artifact_embeds)
    candidate_embeds_list = torch.stack([torch.stack(embeds) for embeds in candidate_embeds_list])
    labels = torch.stack(labels)
    masks = torch.stack(masks)

    # print('times:', times.size())
    # print('event_types:', event_types.size())
    # print('feedbacks:', feedbacks.size())
    # print('artifact_embeds:', artifact_embeds.size())
    # print('candidate_embeds_list:', candidate_embeds_list.size())
    # print('labels_list:', labels.size())
    # print('masks:', masks.size())

    if is_inference:  # 推理模式下，直接按tuple返回数据，方便后续处理
        return (
            times,
            event_types,
            feedbacks,
            artifact_embeds,
            candidate_embeds_list,
            torch.zeros_like(artifact_embeds),
            torch.tensor(1).unsqueeze(0),
        )

    # 计算真正设置label的点的索引
    pred_point_idxs = set_pred_point(masks)
    seqs = []
    max_seq_len = model_params['max_seq_len']

    for pred_idx in pred_point_idxs:
        start_idx = max(0, pred_idx - max_seq_len + 1)
        idx = range(start_idx, pred_idx + 1)  # 包含当前时间步

        if len(idx) > 0:
            first_tp = times[idx][0]
            seqs.append(
                (
                    times[idx] - first_tp,  # 相对时间
                    event_types[idx],
                    feedbacks[idx],
                    artifact_embeds[idx],
                    candidate_embeds_list[idx],
                    labels[idx],
                )
            )
    return seqs


def set_pred_point(masks):
    contiguous_idxs = []
    cross_artifact_idxs = []

    for i, mask in enumerate(masks):
        if mask == torch.tensor(2, device=device).unsqueeze(0):
            cross_artifact_idxs.append(i)
        elif mask == torch.tensor(1, device=device).unsqueeze(0):
            contiguous_idxs.append(i)

    # 按train_params['pred_point_proportion']中的比例设置pred_point_idxs
    pred_point_idxs = []
    proportions = train_params['pred_point_proportion']
    contiguous_prop, cross_artifact_prop = proportions

    len_contiguous = len(contiguous_idxs)
    len_cross_artifact = len(cross_artifact_idxs)

    # 计算各类型能取的最大批次数（避免除零）
    x = 0
    try:
        x = min(
            len_contiguous // contiguous_prop if contiguous_prop > 0 else float('inf'),
            len_cross_artifact // cross_artifact_prop if cross_artifact_prop > 0 else float('inf')
        )
    except:
        raise ValueError('预测点占比设置不合理，或数据量不足')

    # 按比例计算实际取的数量
    contiguous_count = contiguous_prop * x
    cross_artifact_count = cross_artifact_prop * x

    # 随机采样
    if 0 < contiguous_count <= len(contiguous_idxs):
        pred_point_idxs.extend(random.sample(contiguous_idxs, contiguous_count))
    if 0 < cross_artifact_count <= len(cross_artifact_idxs):
        pred_point_idxs.extend(random.sample(cross_artifact_idxs, cross_artifact_count))

    print(
        f'set {len(pred_point_idxs)} pred points, including {contiguous_count} contiguous and {cross_artifact_count} cross-artifact')
    return pred_point_idxs


def concat_json_from_folder(json_folder: str, date_filter: datetime = None) -> List[dict]:
    # 读取train_data_folder下的所有json文件名list
    train_json_list = [file_name for file_name in os.listdir(json_folder) if file_name.endswith('.json')]

    # 文件命名模式：v0.3.0_2025-04-20-16.05.54.029_train.json
    # 其中v0.3.0为版本号，2025-04-20-16.05.54.029为时间戳，train为数据类型，按照时间戳从早到晚排序
    train_json_list.sort(key=lambda x: os.path.basename(x).split('_')[1])
    if date_filter is not None:
        # 过滤出距今date_filter以内的数据
        train_json_list = [file_name for file_name in train_json_list if
                           datetime.strptime(os.path.basename(file_name).split('_')[1],
                                             '%Y-%m-%d-%H.%M.%S.%f') >= date_filter]

    # 读取json文件内容，并将其转换为list
    ret = []
    for file_name in train_json_list:
        with open(os.path.join(json_folder, file_name), 'r', encoding='utf-8') as f:
            ret.extend(json.load(f))

    return ret
