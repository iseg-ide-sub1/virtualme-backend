import argparse
import json
import os
import pickle

import os
import sys
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from base import *

dataset_raw_dir = 'dataset_raw'
dataset_dir = 'dataset'

log = Log(set(), set(), [])


def add_to_artifact_history(artifact: Artifact):
    if artifact is None:
        raise ValueError('Artifact is None')
    for a in log.artifact_history:
        if a.name == artifact.name:
            a.add_count()
            return
    log.artifact_history.add(artifact)


def add_to_cmd_history(cmd: Context):
    if cmd is None:
        raise ValueError('Context is None')
    for c in log.cmd_history:
        if c.get_cmd() == cmd.get_cmd():
            c.add_count()
            return
    log.cmd_history.add(cmd)


def conv_event_type(event_type_raw):
    for e in EventType:
        if e.value == event_type_raw:
            return e
    raise ValueError('Invalid event type: {}'.format(event_type_raw))


def conv_artifact_type(artifact_type_raw):
    for a in ArtifactType:
        if a.value == artifact_type_raw:
            return a
    raise ValueError('Invalid artifact type: {}'.format(artifact_type_raw))


def conv_context_type(context_type_raw):
    for c in ContextType:
        if c.value == context_type_raw:
            return c
    raise ValueError('Invalid context type: {}'.format(context_type_raw))


def conv_task_type(task_type_raw):
    for t in TaskType:
        if t.value == task_type_raw:
            return t
    raise ValueError('Invalid task type: {}'.format(task_type_raw))


def conv_artifact(artifact_raw, reference_raw):
    artifact_type = conv_artifact_type(artifact_raw['type'])

    name = ''
    if 'hierarchy' in artifact_raw:
        for a in artifact_raw['hierarchy']:
            name += a['name'] + '->'
    else:
        name = artifact_raw['name']

    reference = None
    if reference_raw is not None:
        reference = [Artifact]
        for r in reference_raw:
            reference.append(conv_artifact(r, None))

    artifact = Artifact(name, artifact_type, reference)


    return artifact


def conv_context(context_raw):
    context_type = conv_context_type(context_raw['type'])

    content_before = context_raw['content']['before']
    content_after = context_raw['content']['after']
    content = (content_before, content_after)

    start_line = context_raw['start']['line']
    start_character = context_raw['start']['character']
    end_line = context_raw['end']['line']
    end_character = context_raw['end']['character']
    start = (start_line, start_character)
    end = (end_line, end_character)

    context = Context(context_type, content, start, end)

    return context


def preprocess_a_raw(raw_json_file):

    def is_skipped_file_type(file_name):
        for skip_file_type in skipped_file_types:
            if skip_file_type in file_name:
                return True
        return False

    with open(raw_json_file, 'r', encoding='utf-8') as f:
        raw_json = json.load(f)

    raw_num = len(raw_json)

    for item in raw_json:
        id = item['id']
        timestamp = item['timeStamp']
        event_type = conv_event_type(item['eventType'])
        task_type = conv_task_type(item['taskType'])
        artifact = conv_artifact(item['artifact'], item['references'] if 'references' in item else None)
        context = None if 'context' not in item else conv_context(item['context'])

        if is_skipped_file_type(artifact.name):
            continue

        log_item = LogItem(id, timestamp, event_type, task_type, artifact, context)
        log.log_items.append(log_item)
        add_to_artifact_history(artifact)
        if context and context.context_type == ContextType.Terminal:  # 是终端命令执行操作
            add_to_cmd_history(context)

    return raw_num


def preprocess(single_json=None, raw_json_dir=None, pt_name=None):
    dataset_name = ''
    raw_num = 0

    if single_json:
        dataset_name = os.path.basename(single_json).split('.')[0]
        raw_num = preprocess_a_raw(single_json)
    elif raw_json_dir:
        # dataset_name取dir下所有json文件的最大前缀
        files = os.listdir(raw_json_dir)
        files.sort()
        if len(files) > 0:
            dataset_name = pt_name if pt_name else files[0].split('.')[0]
        else:
            raise ValueError('No json files found in the directory.')

        for f in files:
            if f.endswith('.json'):
                raw_num += preprocess_a_raw(os.path.join(raw_json_dir, f))
    else:
        raise ValueError('Please input a single json file or a directory of json files.')

    # 将log.artifact_history和log.cmd_history按照count排序
    log.artifact_history = sorted(log.artifact_history, key=lambda x: x.count, reverse=True)
    log.cmd_history = sorted(log.cmd_history, key=lambda x: x.count, reverse=True)

    # 将log: Log通过pickle序列化后保存到.pt文件中, pickle支持序列化任意python对象，方便保存和再次读取
    with open(f'{dataset_dir}/{dataset_name}.pt', 'wb') as f:
        pickle.dump(log, f)
        print(f'Preprocessed dataset saved to {dataset_dir}/{dataset_name}.pt')

    # print('=====cmd history======')
    # for c in log.cmd_history:
    #     print(c)
    print('=====artifact history======')
    for a in log.artifact_history:
        print(a)

    print('=====clean num======\n', len(log.log_items))
    print('=====raw num======\n', raw_num)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--single-json', type=str, default=None, help='input a single log json file')
    parser.add_argument('-f', '--json-dir', type=str, default=None,
                        help='input a directory of log json files')
    args = parser.parse_args()

    if not args.single_json and not args.json_dir:
        raise ValueError('Please input a single json file or a directory of json files.')

    json_dir = os.path.join(dataset_raw_dir, args.json_dir)
    for file in os.listdir(json_dir):
        # 如果包含文件夹名为virtualme-logs，实际是该文件夹下所有json文件
        if 'virtualme-logs' in file:
            json_dir = os.path.join(json_dir, file)
            break

    preprocess(args.single_json, json_dir, pt_name=args.json_dir)
