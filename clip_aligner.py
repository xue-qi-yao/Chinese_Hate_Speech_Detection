from torch import nn

class CLIPAligner(nn.Module):
    def __init__(self, in_feature=512, out_feature=768):
        super().__init__()
        self.clip_aligner = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=int((out_feature+in_feature)/2)),
            nn.Linear(in_features=int((out_feature+in_feature)/2), out_features=out_feature)
        )

    def forward(self, x):
        return self.clip_aligner(x)