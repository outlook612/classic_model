# %%
import torch
from torch import nn
import torch.nn.functional as F
import math

# %%
from torch import Tensor

#将输入的词汇表转换为指定类型的Embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        #vocab_size 词汇表大小          d_model embedding的维度
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)

# %%
#padding_idx的作用


# #@1 不加padding_idx
# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[0, 2, 0, 5]]) 
# # print(input.shape)   [1,4]
# output=embedding(input)
# print(output)
# # print(output.shape)   [1,4,3]

# #@1 padding_idx=0
# embedding = nn.Embedding(10, 3,padding_idx=0)
# input = torch.LongTensor([[0, 2, 0, 5]]) 
# # print(input.shape)   #[1,4]
# output=embedding(input)
# print(output)
# # print(output.shape)   #[1,4,3]
# #padding_idx指定 embedding的索引0当作分隔符使用，无意义
# #torch.LongTensor([[0, 2, 0, 5]]) 创建tensor的过程中遇到了索引0，就自动转换为分隔符


# #@1 padding_idx=0
# embedding = nn.Embedding(10, 3,padding_idx=1)
# print(embedding.weight)
# input = torch.LongTensor([[0, 2, 0, 5]]) 
# # print(input.shape)   #[1,4]
# output=embedding(input)
# print(output)
# # print(output.shape)   #[1,4,3]
# #padding_idx指定 embedding的索引1当作分隔符使用，无意义
# #torch.LongTensor([[0, 2, 0, 5]]) 创建tensor的过程中遇到了索引0，没有识别为分割符1，就填充索引0 [-1.0863,  0.8460, -0.8932]
# # tensor([[-1.0863,  0.8460, -0.8932],
# #         [ 0.0000,  0.0000,  0.0000],
# #         [-0.0726, -0.3325, -0.0206],
# #         [ 0.0678,  0.0226, -0.3586],
# #         [-1.4361,  1.4184, -0.4830],
# #         [-0.6106, -0.5142,  0.3383],
# #         [ 0.7378, -0.7893,  1.3267],
# #         [-0.9398,  1.2132, -0.7667],
# #         [-0.8192,  0.9002,  1.5136],
# #         [-1.3701, -1.3939, -1.9866]], requires_grad=True)
# # tensor([[[-1.0863,  0.8460, -0.8932],
# #          [-0.0726, -0.3325, -0.0206],
# #          [-1.0863,  0.8460, -0.8932],
# #          [-0.6106, -0.5142,  0.3383]]], grad_fn=<EmbeddingBackward0>)



# %%
#构建Position embedding的原理
from PIL import Image
img = Image.open("/home/code/transformer/position_embedding.png")
img.show()
#构造频率因子的原理公式

# %%
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        #初始化一个矩阵
        self.encoding.requires_grad=False
        #位置编码不需要反向传播
        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        #扩维
        _2i=torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]

# %%
# encoding=torch.zeros(10,10,device="cuda")
# pos=torch.arange(0,10,device='cuda')
# pos=pos.float().unsqueeze(dim=1)
# _2i=torch.arange(0,10,step=2,device="cuda").float()
# encoding[:,0::2]=torch.sin(pos/(10000**(_2i/10)))
# encoding[:,1::2]=torch.cos(pos/(10000**(_2i/10)))
# print(encoding)

# tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
#           1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 8.4147e-01,  5.4030e-01,  1.5783e-01,  9.8747e-01,  2.5116e-02,
#           9.9968e-01,  3.9811e-03,  9.9999e-01,  6.3096e-04,  1.0000e+00],
#         [ 9.0930e-01, -4.1615e-01,  3.1170e-01,  9.5018e-01,  5.0217e-02,
#           9.9874e-01,  7.9621e-03,  9.9997e-01,  1.2619e-03,  1.0000e+00],
#         [ 1.4112e-01, -9.8999e-01,  4.5775e-01,  8.8908e-01,  7.5285e-02,
#           9.9716e-01,  1.1943e-02,  9.9993e-01,  1.8929e-03,  1.0000e+00],
#         [-7.5680e-01, -6.5364e-01,  5.9234e-01,  8.0569e-01,  1.0031e-01,
#           9.9496e-01,  1.5924e-02,  9.9987e-01,  2.5238e-03,  1.0000e+00],
#         [-9.5892e-01,  2.8366e-01,  7.1207e-01,  7.0211e-01,  1.2526e-01,
#           9.9212e-01,  1.9904e-02,  9.9980e-01,  3.1548e-03,  1.0000e+00],
#         [-2.7942e-01,  9.6017e-01,  8.1396e-01,  5.8092e-01,  1.5014e-01,
#           9.8866e-01,  2.3884e-02,  9.9971e-01,  3.7857e-03,  9.9999e-01],
#         [ 6.5699e-01,  7.5390e-01,  8.9544e-01,  4.4518e-01,  1.7493e-01,
#           9.8458e-01,  2.7864e-02,  9.9961e-01,  4.4167e-03,  9.9999e-01],
#         [ 9.8936e-01, -1.4550e-01,  9.5448e-01,  2.9827e-01,  1.9960e-01,
#           9.7988e-01,  3.1843e-02,  9.9949e-01,  5.0476e-03,  9.9999e-01],
#         [ 4.1212e-01, -9.1113e-01,  9.8959e-01,  1.4389e-01,  2.2415e-01,
#           9.7455e-01,  3.5822e-02,  9.9936e-01,  5.6786e-03,  9.9998e-01]],
#        device='cuda:0')

# %%
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionalEmbedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)
        #都是调用函数，self的参数都是return

    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)          
        


