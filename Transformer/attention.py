# %%
import torch
import torch.nn as nn

# %%
x=torch.rand(128,32,512)
d_model=512
n_head=8
print(x)

# %%
class MutiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MutiHeadAttention,self).__init__()
        self.n_head=n_head
        #注意力头数
        self.d_model=d_model
        #输入输出特征维度
        ##线性映射层，将输入映射为 Q K W
        self.w_q=nn.Linear(d_model,n_head)
        #     Args:
        # in_features: size of each input sample
        # out_features: size of each output sample
        # bias: If set to ``False``, the layer will not learn an additive bias.
        #     Default: ``True``
        #y=xW+b 数学的线性变换  
        self.w_k=nn.Linear(d_model,n_head)
        self.w_v=nn.Linear(d_model,n_head)
        #输出融合层，将多头输出并合并会d_model
        self.w_combine=nn.Linear(d_model,n_head)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None):
        batch,time,dimension=q.shape       #batch   序列长度    特征维度
        n_d=self.d_model//self.n_head      #每个头的维度=特征维度/头的个数
        #
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)   #q,k,v线性映射成Q,K,V
        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        #view从后往前看，把tensor先变成(sum,)，然后从最后一个维度开始分块，依次往前
        #permute就是维度互换    从左往后对齐序列
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score=q@k.transpose(2,3)/math.sqrt(n_d)
        #Q维度batch,time,self.n_head,n_d        K^T维度batch,time,n_d,self.n_head
        #transpose 单独两个维度进行互换
        #高纬tensor相乘规则：矩阵的最后两维满足矩阵相乘原则，前面的n-2维相同（或者利用广播机制 可以为1）
        #score batch,time,n_head,n_head
        if mask is not None:
            score=score.masked_fill(mask==0,-100000)#将无效位置填充为-100000 mask==0就是前面提到的padding=0
            score=self.softmax(score)@v
            score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)
            #score batch,time,n_head,n_head       
            output=self.w_combine(score)
            #self定义的self.w_combine=nn.Linear(d_model,n_head)         融合多头注意力
            return output


attention=MutiHeadAttention(d_model,n_head)



# %%
out=attention(x,x,x)
print(out)

# %%



# %%


# %%



