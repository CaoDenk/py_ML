from torch  import nn,Tensor
from multiheadAttentionRelative import MultiheadAttentionRelative

class TransformerSelfAttentionLayer(nn.Mudule):
    """_summary_
    自注意机制
    Args:
        nn (_type_): _description_
    """
    def __init__(self,hidden_dim:int, nhead:int) -> None:
        super().__init__()
        # self.self_attn=  #多头自注意
        self.self_attn=MultiheadAttentionRelative(hidden_dim,nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        s
        
        
        
    def forward(self,feat:Tensor,pos=None,pos_indexes=None):
        """_summary_

        Args:
            feat (Tensor): _description_  图像特征 feature
            pos (_type_, optional): _description_. Defaults to None.
            pos_indexes (_type_, optional): _description_. Defaults to None.
        """
        
        feat2=self.norm1(feat)
        
        feat2,attn_weight ,_ =self.self_attn(query=feat2,key=feat2,vaule=feat2,pos_enc=pos,pos_indexes=pos_indexes)
        
        feat = feat + feat2
        
        return feat
    
    
        