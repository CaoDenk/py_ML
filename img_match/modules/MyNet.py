from img_match.modules.NerfPosiontionalEncoding import NerfPosiontionalEncoding
from img_match.modules.NestedTensor import NestedTensor
from img_match.modules.mlp import MLP
from torch import nn as nn
import torch

class MyNet(nn.Module):

    def __init__(self, backbone, transformer, sine_type='lin_sine'):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.corr_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_proj = NerfPosiontionalEncoding(hidden_dim // 4, sine_type)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, samples: NestedTensor, queries):

        features, pos = self.backbone(samples)
        # samples 1 3 256 512
        #src 1 1024  16  32
        src, mask = features[-1].decompose()
        assert mask is not None
        _b, _q, _ = queries.shape
        queries = queries.reshape(-1, 2)
        queries = self.query_proj(queries).reshape(_b, _q, -1)
        queries = queries.permute(1, 0, 2)
        hs = self.transformer(self.input_proj(src), mask, queries, pos[-1])[0]
        outputs_corr = self.corr_embed(hs)
        out = {'pred_corrs': outputs_corr[-1]}
        return out
    
    
if __name__=='__main__':
    ...