from pathlib import Path
import pandas as pd
import torch 


def load_text(text_path_list):
    text_list = []
    for text_path in text_path_list:
        list_text = read_csv_tsv_aligner(text_path)
        text_list.extend(list_text)
    return text_list


def read_csv_tsv_aligner(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        list_text = list(df["text"])
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        list_text = list(df["content"])
    return list_text


def read_csv_tsv_expert(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        list_text = list(df["text"])
        label = list(df["label"])
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        list_text = list(df["content"])
        label = list(df["toxic"])
    return list_text, label


def downsample(tensor, factor):
    batch_size, seq_len, num_features = tensor.shape
    kernel = torch.ones(num_features, 1, factor, device=tensor.device) / factor
    tensor = tensor.permute(0, 2, 1)
    moving_avg = torch.nn.functional.conv1d(
        tensor, kernel, stride=factor, padding=0, groups=num_features
    )
    return moving_avg.permute(0, 2, 1)


def load_aligner(model, path):
    saved = torch.load(path)
    model.load_state_dict(saved)