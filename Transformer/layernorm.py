# %%
import torch
import torch.nn as nn

# %% [markdown]
# ![](Layernorm.png)

# %%
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-12):
        #eps 稳定性参数
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        #缩放参数 全1
        self.beta=nn.Parameter(torch.zeros(d_model))
        #偏移参数 全0
        self.eps=eps
        #数值稳定性，避免出现除以0的情况
    
    def forward(self,x):
        #keepdim保持维度对齐
        mean=x.mean(-1,keepdim=True)
        #x(128,32,512)  计算512维度的均值
        var=x.var(-1,unbiased=False,keepdim=True)
        #计算有偏方差公式的方差  不是标准差，是方差
        out=(x-mean)/torch.sqrt(var+self.eps)
        # 标准正态
        out=self.gamma*out+self.beta
        #仿射变换
        return 0

# %%



