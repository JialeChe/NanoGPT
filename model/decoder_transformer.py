import torch
import torch.nn as nn
import torch.nn.functional as F
from .component import mha,rmsnorm,moe,ffn_swiglu
from torchsummary import summary

class transformer_block(nn.Module):
    def __init__(self,d_model,n_head,d_hidden,dropout=0.1,use_moe=True,n_expert=4,top_k=2,max_seq_len=512):
        super(transformer_block,self).__init__()
        self.norm1 = rmsnorm(d_model)
        self.norm2 = rmsnorm(d_model)
        self.mha = mha(n_head,d_model,max_seq_len=max_seq_len)
        if use_moe:
            self.ffn = moe(n_expert,d_model,top_k,dropout)
        else:
            self.ffn = ffn_swiglu(d_model,d_hidden,dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        # x: [B,L,d_model]
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm,x_norm,x_norm,mask)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x
    
class decoder_transformer(nn.Module):
    
    
    def __init__(self,vocab_size,d_model,n_layer,n_head,d_hidden,dropout=0.1,
                 use_moe=True,n_expert=4,top_k=2,max_seq_len=512):
        super(decoder_transformer,self).__init__()
        self.token_emb = nn.Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
            transformer_block(d_model,n_head,d_hidden,dropout,use_moe,n_expert,top_k,max_seq_len)
            for _ in range(n_layer)
        ])
        self.norm = rmsnorm(d_model)
        self.head = nn.Linear(d_model,vocab_size,bias=False)
        self.max_seq_len = max_seq_len
    
    def forward(self,inputs):
        # inputs: [B,L]
        if inputs.dtype != torch.long:
            inputs = inputs.long()
        b,l = inputs.shape
        assert l <= self.max_seq_len,f"序列长度超过模型最大长度限制"

        x = self.token_emb(inputs) # [B,L,d_model]
        mask = torch.tril(torch.ones((l,l),device=inputs.device)).unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

        for layer in self.layers:
            x = layer(x,mask)

        x = self.norm(x)
        logits = self.head(x)  # [B,L,vocab_size]

        return logits
    
if __name__ == "__main__":
    model = decoder_transformer(vocab_size=100277,d_model=512,n_layer=8,n_head=8,d_hidden=1024,
                                dropout=0.1,use_moe=True,n_expert=4,top_k=2,max_seq_len=512).cuda()
    inputs = torch.randint(0,100277,(64,256)).cuda()
    logits = model(inputs)
    print(logits.shape)
    summary(model,(256,),batch_size=2)