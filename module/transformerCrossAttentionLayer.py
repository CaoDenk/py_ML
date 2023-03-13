import torch
import torch.nn as nn
import torch.nn.functional as F
from module.transformer import Transformer
import multiheadAttentionRelative

class TransformerCrossAttentionLayer(nn.Module):
    
    def __init__(self,hidden_dim:int,nhead:int) -> None:
        super().__init__()
        
        self.cross_attn=multiheadAttentionRelative(hidden_dim,nhead)
        
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
        
    def forward(self,feat_left,feat_right,pos:None,pos_indexes:None,last_layer=False):
        
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm2(feat_right)
        
        if pos is not None:
            pos_flipped = torch.flip(pos,[0])
        else:
            pos_flipped = pos
        feat_right_2 = self.cross_attn(query=feat_right_2,key=feat_left_2,value=feat_left_2,pos_enc=pos_flipped,pos_indexes=pos_indexes)[0]
        
        feat_right =feat_right+feat_right_2
        
        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)
        else:
            attn_mask=None
            
            
        #正则化feat_right
        feat_right_2=self.norm2(feat_right)
        feat_left_2,attn_weight,raw_attn= self.cross_attn(query=feat_left_2,key=feat_right,value=feat_right_2,attn_mask=attn_mask,pos_enc=pos,pos_indexes=pos_indexes)
        
        
        feat_left=feat_left+feat_left_2
        
        feat = torch.cat([feat_left,feat_right],dim=1)
        return feat,raw_attn
    
    
    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular

        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask


    def build_transformer(args):
        return Transformer( hidden_dim=args.channel_dim,nhead=args.nheads,num_attn_layers=args.num_attn_layers)