"""
单请求、无 padding、预先分配KV的缓存空间避免多次复制
即假定每批次输出的长度都相同

实现RoPE:相邻两维度组成一对进行旋转
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head dim 必须是偶数")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        """
        生成公式中的(base**2i/d)**(-1)这个序列:
            torch.arrage(0, head_dim, 2)生成偶数索引 [0,2,4,....head_dim-2]
            .float()转为浮点数
            / head_dim归一化到0-1之间
        """
        angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        """
        torch.arange(max_seq_len).float()生成位置索引转为浮点数运算
        torch.outer(a, b)[i, j] = a[i] * b[j]即位置乘以旋转角度m 与 \theta的外积生成一个矩阵
        """
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):  # [B,H,T,head_dim]
        cos = self.cos[positions].to(x.dtype)[None, None, :, :]
        sin = self.sin[positions].to(x.dtype)[None, None, :, :]

        even, odd = x[..., 0::2], x[..., 1::2]

        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)


class LayerKVCache:
    def __init__(self, k_new, v_new, capacity):
        B, H, _, D = k_new.shape
        self.k = k_new.new_empty(B, H, capacity, D)
        self.v = v_new.new_empty(B, H, capacity, D)
        self.length = 0

    def append(self, k_new, v_new):
        start = self.length
        end = start + k_new.size(-2)
        if end > self.k.size(-2):
            raise ValueError("KV CACHE is not enough")

        self.k[:, :, start:end, :].copy_(k_new)
        self.v[:, :, start:end, :].copy_(v_new)
        self.length = end

        return self.k[:, :, :end, :], self.v[:, :, :end, :]


class CasualSelfAttention(nn.Module):
    """
    Q/K/V 投影、拆分多头、因果掩码、注意力计算、合并多头、输出投影
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.d_head = config["d_model"] // config["n_heads"]
        self.Wq = nn.Linear(config["d_model"], config["d_model"])
        self.Wk = nn.Linear(config["d_model"], config["d_model"])
        self.Wv = nn.Linear(config["d_model"], config["d_model"])
        self.ffn = nn.Linear(config["d_model"], config["d_model"])
        self.drop = nn.Dropout(config["dropout"])

        self.RoPE = RotaryEmbedding(self.d_head, config["max_seq_len"])

        self.register_buffer(
            "mask", torch.tril(torch.ones(config["max_seq_len"], config["max_seq_len"]))
        )

    def mutihead(self, x):
        B, T, C = x.shape

        assert C % self.config["n_heads"] == 0

        x = x.reshape(B, T, self.config["n_heads"], self.d_head)
        x = x.transpose(1, 2)

        return x

    def forward(self, x, positions, past_key_value=None, use_cache=False):
        B, T, C = x.shape  # T表示本轮输入的token数目

        Q = self.mutihead(self.Wq(x))  # [B,H, T,d_head]
        K_new = self.mutihead(self.Wk(x))  # 本次输入新生成的KV
        V_new = self.mutihead(self.Wv(x))
        Q = self.RoPE(Q, positions)
        K_new = self.RoPE(K_new, positions)

        past_len = 0

        if use_cache:
            cache = past_key_value
            if cache is None:  # 第一次prefill
                cache = LayerKVCache(K_new, V_new, self.config["max_seq_len"])
            past_len = cache.length

            K, V = cache.append(K_new, V_new)
            present = cache
        else:
            past_len = 0
            K, V = K_new, V_new
            present = None


        total_len = past_len + T

        score = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)  # [B,H, T,total_len]
        # 修改mask
        casual_mask = self.mask[past_len:total_len, :total_len]  # [T,total_len]
        score = score.masked_fill(casual_mask == 0, float("-inf"))

        score = F.softmax(score, dim=-1)
        score = self.drop(score)

        score = score @ V  # [B,H, T,d_head]
        score = score.transpose(1, 2)  # [B,T, H,d_head]
        output = score.reshape(B, T, self.config["d_model"])
        output = self.ffn(output)

        return output, present


class TransformerBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.atte = CasualSelfAttention(config)
        self.layerNorm1 = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.layerNorm2 = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_ff"]),
            nn.GELU(),
            nn.Linear(config["d_ff"], config["d_model"]),
        )
        self.drop = nn.Dropout(config["dropout"])

    def forward(self, x, positions, past_key_value=None, use_cache=False):

        attn_output, present = self.atte(
            self.layerNorm1(x), positions, past_key_value, use_cache
        )

        x = x + self.drop(attn_output)
        x = x + self.drop(self.ffn(self.layerNorm2(x)))
        return x, present


class LiteGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            self.config["vocab_size"], config["d_model"]
        )
        self.trans = nn.ModuleList(
            TransformerBlock(config) for _ in range(config["n_layers"])
        )
        self.layerNorm = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"])
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

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False):
        """targets 与 input_ids 同形状，由调用方提前错开一个 token。"""
        B, T = input_ids.shape
        if not 0 < T <= self.config["max_seq_len"]:
            raise ValueError("输入长度必须在 1 和 max_seq_len 之间")

        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets 必须与 input_ids 形状相同")

        if past_key_values is None:
            past_len = 0
            past_key_values = (None,) * len(self.trans)
        else:
            past_len = past_key_values[0].length

        if T <= 0 or past_len + T > self.config["max_seq_len"]:
            raise ValueError("输入长度超过最大限制")

        positions = torch.arange(past_len, past_len + T, device=input_ids.device)
        x = self.token_embedding(input_ids)
        x = self.drop(x)

        presents = []

        for index, block in enumerate(self.trans):
            x, present = block(
                x,
                positions=positions,
                past_key_value=past_key_values[index],
                use_cache=use_cache,
            )
            if use_cache:
                presents.append(present)

        x = self.layerNorm(x)
        logits = self.lm_head(x)  # [B,T,vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss, tuple(presents) if use_cache else None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50):
        B, T = input_ids.shape
        cache = None
        current_input = input_ids

        max_seq_len = self.config["max_seq_len"]

        self.eval()

        for _ in range(max_new_tokens):

            logits, _, cache = self(
                current_input,  # 这样写,第一轮后每次输入只有新增的一个token
                past_key_values=cache,
                use_cache=True,
            )

            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            next_token_id = next_token_logits.argmax(
                dim=-1,
                keepdim=True,
            )  # [B, 1]
            current_input = next_token_id  # 这样写,第一轮后每次输入只有新增的一个token
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
