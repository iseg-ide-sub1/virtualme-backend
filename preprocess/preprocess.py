import argparse
import json

from base import Artifact


def get_history_list(raw_json):
    op_artifact_list = []
    ref_artifact_list = []
    cmd_list = []

    for log_item in raw_json:
        if 'artifact' in log_item:
            name = log_item['name']
            artifact_type = log_item['type']


    return op_artifact_list, ref_artifact_list, cmd_list


def preprocess_raw(raw_json_file):
    with open(raw_json_file, 'r') as f:
        raw_json = json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='input a single log json file')
    parser.add_argument('-o', type=str, help='specialize the output json name')
    args = parser.parse_args()

    preprocess_raw(args.i)

