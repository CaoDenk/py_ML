import torch
from typing import Optional
class NestedTensor():
    def __init__(self,tensor,mask:Optional[torch.Tensor]) -> None:
        self.tensor=tensor
        self.mask=mask
        
    def to(self,device):
        cast_tensor=self.tensor.to(device)
        mask=self.mask
        if mask is not None:
            cast_mask=mask.to(device)
            return NestedTensor(cast_tensor,cast_mask)
        else:
            return NestedTensor(cast_tensor,None)
        
        