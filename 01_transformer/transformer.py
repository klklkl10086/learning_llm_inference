import math
import torch
import torch.nn as nn
from torch.nn import functional as F




class CasualSelfAttention(nn.Module):
    """
    Q/K/V 投影、拆分多头、因果掩码、注意力计算、合并多头、输出投影
    """
    def __init__(self,config:dict):
        super().__init__()
        self.config = config
        self.d_head = config["d_model"]//config["n_heads"]
        self.Wq = nn.Linear(config["d_model"],config["d_model"])
        self.Wk = nn.Linear(config["d_model"],config["d_model"])
        self.Wv = nn.Linear(config["d_model"],config["d_model"])
        self.ffn = nn.Linear(config["d_model"],config["d_model"])
        self.drop = nn.Dropout(config["dropout"])

        self.register_buffer('mask',torch.tril(torch.ones(config["max_seq_len"],config["max_seq_len"])))

    def mutihead(self,x):
        B,T,C = x.shape

        assert C%self.config["n_heads"]==0

        x = x.reshape(B,T,self.config["n_heads"],self.d_head)
        x = x.transpose(1,2)

        return x

    
    def forward(self,x):
        B,T,C = x.shape
        Q = self.mutihead(self.Wq(x))# [B,H, T,d_head]
        K = self.mutihead(self.Wk(x))
        V = self.mutihead(self.Wv(x))


        score = Q @ K.transpose(-2,-1) / math.sqrt(self.d_head)# [B,H, T,T]
        score = score.masked_fill(self.mask[:T, :T] == 0, float('-inf'))# [B,H, T,T]
        score = F.softmax(score,dim=-1)# [B,H, T,T]
        score = self.drop(score)
        score = score @ V# [B,H, T,d_head]
        score = score.transpose(1,2)# [B,T, H,d_head]
        output = score.reshape(B,T,self.config["d_model"])
        output = self.ffn(output)

        return output
        



class TransformerBlock(nn.Module):
    def __init__(self,config:dict):
        super().__init__()
        self.atte = CasualSelfAttention(config)
        self.layerNorm1=nn.LayerNorm(config["d_model"],eps=config["layer_norm_eps"])
        self.layerNorm2=nn.LayerNorm(config["d_model"],eps=config["layer_norm_eps"])
        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_ff"]),
            nn.GELU(),
            nn.Linear(config["d_ff"], config["d_model"]),
            )
        self.drop = nn.Dropout(config["dropout"])
        
    def forward(self,x):
        x = x + self.drop(self.atte(self.layerNorm1(x)))
        x = x + self.drop(self.ffn(self.layerNorm2(x)))
        return x


class LiteGPT(nn.Module):
    def __init__(self,config:dict):
        super().__init__()
        self.config=config
        self.token_embedding = nn.Embedding(self.config["vocab_size"],config["d_model"])
        self.position_embedding = nn.Embedding(config["max_seq_len"], config["d_model"])
        self.trans=nn.ModuleList(TransformerBlock(config) for _ in range(config["n_layers"]))
        self.layerNorm=nn.LayerNorm(config["d_model"],eps=config["layer_norm_eps"])
        self.lm_head = nn.Linear(config["d_model"],config["vocab_size"])
        self.drop = nn.Dropout(config["dropout"])

        self.apply(self._init_weights)
        # 每层有两个残差分支，适当缩小输出投影的初始尺度。
        residual_std = 0.02 / math.sqrt(2 * config["n_layers"])
        for block in self.trans:
            nn.init.normal_(block.atte.ffn.weight, std=residual_std)
            nn.init.normal_(block.ffn[-1].weight, std=residual_std)

        if config["tie_embeddings"]:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self,input_ids,targets=None):
        """targets 与 input_ids 同形状，由调用方提前错开一个 token。"""
        B,T = input_ids.shape
        if not 0 < T <= self.config["max_seq_len"]:
            raise ValueError("输入长度必须在 1 和 max_seq_len 之间")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets 必须与 input_ids 形状相同")
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for trans in self.trans:
            x = trans(x)
        x=self.layerNorm(x)
        logits = self.lm_head(x)  # [B,T,vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=50):
        B,T = input_ids.shape
        max_seq_len = self.config["max_seq_len"]
        for _ in range(max_new_tokens):
            context_ids = input_ids[:, -max_seq_len:]
            logits, _ = self(context_ids)
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            next_token_id = next_token_logits.argmax(
                dim=-1,
                keepdim=True,
            )  # [B, 1]
            input_ids = torch.cat(
                [input_ids, next_token_id],
                dim=1,
            )
        return input_ids


def get_config(vocab_size):
    return {
        "vocab_size": vocab_size,
        "max_seq_len": 256,
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 4,
        "d_ff": 1024,
        "dropout": 0.1,
        "layer_norm_eps": 1e-5,
        "tie_embeddings": True,
    }


