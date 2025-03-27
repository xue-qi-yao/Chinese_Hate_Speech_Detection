from pathlib import Path
import pandas as pd
import torch 

def read_csv_tsv(path):
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        list_text = list(df["text"])
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
        list_text = list(df["content"])
    return list_text


def downsample(tensor, factor):
    batch_size, seq_len, num_features = tensor.shape
    kernel = torch.ones(num_features, 1, factor, device=tensor.device) / factor
    tensor = tensor.permute(0, 2, 1)
    moving_avg = torch.nn.functional.conv1d(
        tensor, kernel, stride=factor, padding=0, groups=num_features
    )
    return moving_avg.permute(0, 2, 1)