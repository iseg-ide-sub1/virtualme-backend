import os

try:
    from .config import *
    from .encoder import encode_from_json, concat_json_from_folder
    from .CfC import CFC
except ImportError:
    from config import *
    from encoder import encode_from_json, concat_json_from_folder
    from CfC import CFC


def inference(input_json):
    max_seq_len = model_params['max_seq_len']
    seq = encode_from_json(input_json, is_inference=True)
    if seq[0].__len__() > max_seq_len:
        raise ValueError(f"Sequence length is longer than {max_seq_len}")

    ckpt_path = os.path.join(os.path.dirname(__file__), 'ckpt', 'cfc', 'version_1', 'cfc-best.ckpt')
    cfc = CFC.load_from_checkpoint(ckpt_path)
    cfc.eval()

    candidates = input_json[-1]['candidates']
    output = cfc.inference(seq, candidates, k=model_params['k'])
    return output


if __name__ == '__main__':
    input_json = concat_json_from_folder('../dataset_raw')
    pred_artifacts = inference(input_json)
    print(pred_artifacts)
