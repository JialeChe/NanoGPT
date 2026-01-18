import torch
from torch import nn
import torch.nn.functional as F
import math
from .utils import apply_rotary_emb,precompute_freqs_cis

class mha(nn.Module):
    
    def __init__(self,n_head,d_model,max_seq_len=512):
        super(mha, self).__init__()

        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head

        self.w_k = nn.Linear(d_model,d_model)
        self.w_q = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_conbine = nn.Linear(d_model,d_model,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(dim=d_model//n_head, end=max_seq_len)
    
    def forward(self,q,k,v,mask=None):
        b,t,d = q.shape #  (batch,time,d_model)
        assert d == self.d_model,f"d_model can't // n_head completely"
        n_d = self.d_model // self.n_head

        q,k,v = self.w_q(q),self.w_k(k),self.w_v(v)
        q = q.view(b,t,self.n_head,n_d) # shape: (b,t,n_head,n_d)
        k = k.view(b,t,self.n_head,n_d)
        v = v.view(b,t,self.n_head,n_d).transpose(1,2) # shape: (b,n_head,t,n_d)
        # apply rope
        freqs_cos = self.freqs_cos[:t,:].to(q.device)
        freqs_sin = self.freqs_sin[:t,:].to(q.device)
        q,k = apply_rotary_emb(q,k,freqs_cos,freqs_sin)
        
        q = q.transpose(1,2) # shape: (b,n_head,t,n_d)
        k = k.transpose(1,2)
        
        attn = q @ k.transpose(-2,-1) / math.sqrt(n_d) # shape: (b,n_head,t,t)

        if mask is not None:
            attn = attn.masked_fill(mask==0,float('-inf')) # mask==0 -> -inf
        
        result = self.softmax(attn)@v
        result = result.transpose(1,2).contiguous().view(b,t,d)
        result = self.w_conbine(result)
        return result

class rmsnorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super(rmsnorm,self).__init__()
        self.gamma =  nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self,x):
        # Formula: x * 1/sqrt(E[x^2] + eps) * weight
        result = x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps) * self.gamma
        return result

class ffn(nn.Module):
    def __init__(self,d_model,d_hidden,dropout=0.1):
        super(ffn,self).__init__()

        self.fc1 = nn.Linear(d_model,d_hidden)
        self.fc2 = nn.Linear(d_hidden,d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self,x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x)))) 
    
class ffn_swiglu(nn.Module):

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        #第一个 Linear 输出 2 * d_hidden
        self.w_gate_up = nn.Linear(d_model, 2 * d_hidden)  #同时生成 gate 和 up
        self.w_down = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: [B, L, d_model]
        gate_up = self.w_gate_up(x)                     # [B, L, 2 * d_hidden]
        gate, up = gate_up.chunk(2, dim=-1)             # each: [B, L, d_hidden]
        swiglu_out = F.silu(gate) * up                  # [B, L, d_hidden]
        out = self.w_down(self.dropout(swiglu_out))     # [B, L, d_model]
        return out
        
class moe(nn.Module):
    def __init__(self,n_expert,d_model,top_k=2,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_expert = n_expert
        self.top_k = top_k
        self.dropout = dropout

        self.gate = nn.Linear(d_model,n_expert,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.experts = nn.ModuleList([ffn(self.d_model,self.d_model*4,dropout) for _ in range(n_expert)])  relu激活的expert
        self.experts = nn.ModuleList([ffn_swiglu(self.d_model,self.d_model*2,dropout) for _ in range(n_expert)])

    def forward(self,x):
        b,t,d = x.shape
        assert d == self.d_model,f"输入维度和moe设置维度不匹配"

        x_flat = x.view(-1,d)
        N = x_flat.shape[0]

        gate_logits = self.gate(x_flat)
        topk_weights,topk_indices = torch.topk(gate_logits,self.top_k,dim=-1)
        topk_weights= self.softmax(topk_weights)

        out = torch.zeros_like(x_flat)

        for i,expert in enumerate(self.experts):
            mask = (topk_indices == i)
            if not mask.any():
                continue
            token_indices,expert_pos = torch.where(mask)
            select_x = x_flat[token_indices]
            expert_out = expert(select_x)
            weights = topk_weights[token_indices, expert_pos]
            out.index_add_(0, token_indices, expert_out * weights.unsqueeze(1))
        return out.view(b,t,d)