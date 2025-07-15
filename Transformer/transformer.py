#%%
import torch
from torch import nn
import torch.nn.functional as F
import math
from encoder import Encoder
from decoder import Decoder


#%%
class Transformer(nn.Moudle):
    def __init__(self,
                src_pad_idx,trg_pad_idx,    #pad的idx索引
                enc_voc_size,dec_voc_size,  #voc编码器解码器的词表大小
                d_model,
                n_heads,
                ffn_hidden,
                n_layers,
                drop_prob,
                device):
        super(Transformer,self).__init__()

        self.encoder=Encoder(enc_voc_size,
                            max_len,
                            d_model,
                            ffn_hidden,
                            n_head,
                            n_layers,
                            drop_prob,
                            device
                            )
        self.decoder=Decoder(dec_voc_size,
                            max_len,
                            d_model,
                            ffn_hidden,
                            n_heads,
                            n_layers,
                            drop_prob,
                            device,
                            )
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device
    
    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):#q k 是src输入seq       pad是索引序列位置
        len_q,len_k=q.size(1),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3) #not equal  值不等时返回True    找出非填充位 然后在1 3维度扩容
        q=q.repeat(1,1,1,len_k)                     #沿最后一个维度复制len_k次
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1)
        mask=q&k
        return mask

    def mask_casual_mask(self,q,k):
        mask=torch.trill(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)*self.make_casual_mask(trg,trg)
        enc=self.encoder(src,src_mask)
        out=self.decocer(trg,src,trg_mask,src_mask)
        return out





 



