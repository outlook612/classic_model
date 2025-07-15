# %%
import torch
import torch.nn as nn
%run attention.ipynb


# %%
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
#就是Feed Forward模块的内部构造

# %% [markdown]
# ![](structure.png)

# %%
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head)
        #注意力层
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PostionwiseFeedForward(d_model,ffn_hidden,dropout)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        _x=x
        #残差
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x
#EncoderLayer完全按paper(见上图)来构造的

# %%
class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,dropout=0.1,device):
        #enc_voc_size词汇表大小 
        super(Encoder,self).__init__()
        self.embedding=TransformerEmbedding(enc_voc_size,max_len,d_model,dropout=0.1,device)
        self.layers=nn.ModuleList(
            [
                EncoderLayer(d_model,ffn_hidden,n_head,device)
                for _ in range(n_layers)
            ]
        )

    def forward(self,x,s_mask):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask)
        return x

# %%



