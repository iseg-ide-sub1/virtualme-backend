import json
import os

import torch

try:
    from .config import *
    from .encoder import encode_from_json, concat_json_from_folder
except ImportError:
    from config import *
    from encoder import encode_from_json, concat_json_from_folder


class EventDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.seqs = []

    def load_train_data_from_raw(self, train_data_folder):
        # 读取train_data_folder下的所有json文件名list
        train_json_list = concat_json_from_folder(train_data_folder)

        self.seqs = encode_from_json(train_json_list)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

    def __repr__(self):
        ret = f'EventDataset({len(self.seqs)} sequences)\n'
        if len(self.seqs) == 0:
            return ret
        ret += f'Each sequence has max length {self.seqs[0][0].size(0)} events\n'
        ret += f'Each event has {len(self.seqs[0])} features, including time, event_type, feedback, artifact_embed, candidate_embeds, labels and mask\n'
        times, event_types, feedbacks, artifact_embeds, candidate_embeds_list, labels, masks = self.seqs[0]
        ret += f'For example, an event has the following features:\n'
        ret += f'Time: 1 dim tensor\n'
        ret += f'Event type: {event_types.size(1)} dim tensor\n'
        ret += f'Feedback: 1 dim tensor\n'
        ret += f'Artifact embed: {artifact_embeds.size(1)} dim tensor\n'
        ret += f'In-feature total: {event_types.size(1) + 1 + artifact_embeds.size(1)} dim tensor\n'
        ret += f'Candidate embeds: {candidate_embeds_list.size(1)} 个 {candidate_embeds_list.size(2)} dim tensor\n'
        ret += f'Label: {labels.size(1)} dim tensor\n'
        ret += f'Mask: 1 dim tensor\n'
        ret += f'Model params: {model_params}\n'

        return ret


if __name__ == '__main__':
    event_dataset = EventDataset()
    event_dataset.load_train_data_from_raw("../dataset_raw")
    print(event_dataset)
