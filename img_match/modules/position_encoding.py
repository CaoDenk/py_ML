import torch
import torch.nn as nn
import NerfPosiontionalEncoding
# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on images.
#     """
#     def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine'):
#         super().__init__()
#         self.num_pos_feats = num_pos_feats
#         # self.temperature = temperature
#         self.normalize = normalize
#         self.sine = NerfPositionalEncoding(num_pos_feats//2, sine_type)

#     @torch.no_grad()
#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         mask = tensor_list.mask
#         assert mask is not None
#         not_mask = ~mask
#         y_embed = not_mask.cumsum(1, dtype=torch.float32)
#         x_embed = not_mask.cumsum(2, dtype=torch.float32)
#         eps = 1e-6
#         y_embed = (y_embed-0.5) / (y_embed[:, -1:, :] + eps)
#         x_embed = (x_embed-0.5) / (x_embed[:, :, -1:] + eps)
#         pos = torch.stack([x_embed, y_embed], dim=-1)
#         return self.sine(pos).permute(0, 3, 1, 2)


class PositionEmbeddingSine(nn.Module):
    def __init__(self,num_pos_feat=64,normalize=False,scale=None,sine_type='lin_sine') -> None:
        super().__init__()
        # self.normalize=normalize
        self.sine=NerfPosiontionalEncoding(num_pos_feat//2,sine_type)
    @torch.no_grad()
    def forward(self):
        ...
        
        
