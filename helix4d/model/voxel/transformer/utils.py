from torch import nn

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def feed_forward(dim_input: int = 512, mul_feedforward: int = 1, drop=.25) -> nn.Module:

    if mul_feedforward == 0:
        ffn = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.Dropout(drop)
        )
    else:
        ffn = nn.Sequential(
            nn.Linear(dim_input, mul_feedforward * dim_input),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mul_feedforward * dim_input, dim_input),
            nn.Dropout(drop)
        )

    ffn.apply(_init_weights)
    return ffn