"""
仅支持不同长度请求批量decode
新增RequestState类管理请求
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class RequestState:
    def __init__(self, token_ids, past_key_values):
        if not token_ids or not past_key_values:
            raise ValueError("需要非空prompt和prefill后的KV cache")

        self.token_ids = list(token_ids)
        self.past_key_values = tuple(past_key_values)

        # 创建状态时应已全部prefill
        if self.cached_len != len(self.token_ids):
            raise ValueError("初始化时token_ids应恰好对应已缓存的prompt")

    @property  # 把方法伪装成属性，让你能像访问属性一样调用它
    def cached_len(self):
        lengths = [cache.length for cache in self.past_key_values]
        if len(set(lengths)) != 1:
            raise ValueError("同一请求的各层缓存长度不一致")
        return lengths[0]

    @property
    def pending_token(self):  # 最新要进入decode的token
        if len(self.token_ids) != self.cached_len + 1:
            raise ValueError("每个请求必须恰有一个尚未计算的token")
        return self.token_ids[-1]

    def append_token(self, token_id):  # 最新decode完成的token
        if len(self.token_ids) != self.cached_len:
            raise ValueError("先计算上一个待处理 token，再追加新 token")
        self.token_ids.append(int(token_id))


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

    # 兼容单请求和多请求
    def forward(self, x, positions):
        # x:[B,H,T,head_dim]
        # 单请求: position = [T]
        # 多请求: position = [B,T]

        B, _, T, _ = x.shape
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        if (
            positions.ndim != 2
            or positions.shape[1] != T
            or positions.shape[0] not in (1, B)
        ):
            raise ValueError("positions 应为 [T] 或 [B,T]")

        cos = (
            self.cos[positions].to(x.dtype).unsqueeze(1)
        )  # 在第 1 维（dim=1）插入一个长度为 1 的维度
        sin = self.sin[positions].to(x.dtype).unsqueeze(1)

        even, odd = x[..., 0::2], x[..., 1::2]

        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(
            -2
        )  # flatten(start_dim=-2) 表示把最后两维（-2 和 -1）合并成一维：


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

    def decode_batch(self, x, positions, caches):
        B, T, _ = x.shape
        if T != 1 or positions.shape != (B, 1) or len(caches) != B:
            raise ValueError("decode_batch 要求每个请求恰有一个 token/cache")

        Q = self.mutihead(self.Wq(x))  # [B,H,1,D]
        K_new = self.mutihead(self.Wk(x))
        V_new = self.mutihead(self.Wv(x))

        Q = self.RoPE(Q, positions)
        K_new = self.RoPE(K_new, positions)

        # 每个请求有自己的kvcache即LayerKVCache
        for b, cache in enumerate(caches):
            if (
                cache.k.size(0) != 1
            ):  # cache.k 的形状约定为 [1, head数, 缓存容量, head维度]
                raise ValueError("每个 LayerKVCache 应只属于一个请求")
            cache.append(K_new[b : b + 1], V_new[b : b + 1])

        lengths = [cache.length for cache in caches]
        Lmax = max(lengths)

        H, D = Q.size(1), Q.size(-1)

        K = Q.new_zeros(
            B, H, Lmax, D
        )  # 创建一个形状为 (B, H, Lmax, D) 的全零张量，同时自动沿用 Q 的 dtype 和 device（和其他张量属性）。
        V = Q.new_zeros(B, H, Lmax, D)

        for b, cache in enumerate(caches):
            n = lengths[b]
            K[b, :, :n] = cache.k[0, :, :n]  # 作用在前 3 维，第 4 维隐式:
            V[b, :, :n] = cache.v[0, :, :n]

        score = Q @ K.transpose(-2, -1) / math.sqrt(D)  # [B,H,1,Lmax]
        valid = (
            torch.arange(Lmax, device=x.device)[None, :]  # [1, Lmax]
            < torch.tensor(lengths, device=x.device)[:, None]  # [B, 1]
        )  # [B,Lmax]

        score = score.masked_fill(~valid[:, None, None, :], float("-inf"))

        prob = self.drop(F.softmax(score, dim=-1))
        output = (prob @ V).transpose(1, 2).reshape(B, 1, self.config["d_model"])
        return self.ffn(output)


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

    def decode_batch(self, x, positions, caches):
        attn_output = self.atte.decode_batch(self.layerNorm1(x), positions, caches)
        x = x + self.drop(attn_output)
        x = x + self.drop(self.ffn(self.layerNorm2(x)))
        return x


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


    @torch.no_grad()
    def decode_batch(self, requests):
        if not requests:
            raise ValueError("请求列表不能为空")
        if self.training:
            raise ValueError("批量 decode 请先调用 model.eval()")
        for req in requests:
            if len(req.past_key_values) != len(self.trans):
                raise ValueError("请求的 KV cache 层数不匹配")
            if req.cached_len >= self.config["max_seq_len"]:
                raise ValueError("超过 max_seq_len")
        
        #同一个请求不能在 batch 中出现两次 id取对象的"身份标识" {}set
        if len({id(req) for req in requests}) != len(requests):
            raise ValueError("同一个请求不能重复加入 decode batch")

        # 不同请求也不能共用某一层的 KV 缓存
        for layer_idx in range(len(self.trans)):
            cache_ids = [id(req.past_key_values[layer_idx]) for req in requests]
            if len(set(cache_ids)) != len(cache_ids):
                raise ValueError(f"第 {layer_idx} 层的 KV cache 被多个请求共用")

        device = self.token_embedding.weight.device
        positions = torch.tensor(
            [req.cached_len for req in requests],
            dtype=torch.long,
            device=device,
        )[:, None]

        input_ids = torch.tensor(
            [req.pending_token for req in requests], dtype=torch.long, device=device
        )[
            :, None
        ]  # [B,1]

        x = self.drop(self.token_embedding(input_ids))

        for layer_idx, block in enumerate(self.trans):
            layer_caches = [req.past_key_values[layer_idx] for req in requests]
            x = block.decode_batch(x, positions, layer_caches)
        return self.lm_head(self.layerNorm(x))  # [B,1,vocab_size]


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
