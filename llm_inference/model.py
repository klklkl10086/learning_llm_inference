"""仅保留批量 prefill/decode 的推理路径；每个请求持有独立 KV cache。

与 batch_prefill_decode.py 保持相同的模型结构，便于对照代码和加载权重。
prefill 直接把各请求的有效 K/V 写入最终缓存，不再创建整批全容量 KV cache。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class KVCache:
    """一个请求的所有层 KV,length只在完整模型前向结束后推进。"""

    def __init__(self, num_layers: int, capacity: int):
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.capacity = capacity
        self.length = 0 #缓存当前已有多少个token的有效 KV

    def write(self, layer_idx, start_pos, k_new, v_new):
        """写入指定层的新KV不改变整个请求的缓存长度。"""
        end_pos = start_pos + k_new.size(-2)
        if end_pos > self.capacity:
            raise ValueError("KV cache 容量不足")

        if self.keys[layer_idx] is None:
            # 首次写入时按实际 K/V 的设备、dtype 分配，兼容 autocast。
            B, H, _, D = k_new.shape
            self.keys[layer_idx] = k_new.new_empty(B, H, self.capacity, D) #分配空间
            self.values[layer_idx] = v_new.new_empty(B, H, self.capacity, D)

        self.keys[layer_idx][:, :, start_pos:end_pos, :].copy_(k_new)
        self.values[layer_idx][:, :, start_pos:end_pos, :].copy_(v_new)

    def get_layer(self, layer_idx, end_pos):
        """读取指定层的KVcache缓存,包含刚生成的"""
        return (
            self.keys[layer_idx][:, :, :end_pos, :],
            self.values[layer_idx][:, :, :end_pos, :],
        )

    def advance(self, num_tokens: int):
        """更新length"""
        if self.length + num_tokens > self.capacity:
            raise ValueError("KV cache 容量不足")
        self.length += num_tokens


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

    def prefill_batch(self, x, positions, lengths, caches, layer_idx):
        """计算补齐后的整批注意力 只缓存每个请求的有效prompt,不包含padding部分。"""
        B, T, _ = x.shape
        Q = self.RoPE(self.mutihead(self.Wq(x)), positions)
        K_new = self.RoPE(self.mutihead(self.Wk(x)), positions)
        V_new = self.mutihead(self.Wv(x))

        # prefill无历史KV；直接用新算出的 K/V 计算注意力。
        score = Q @ K_new.transpose(-2, -1) / math.sqrt(self.d_head)
        causal_mask = self.mask[:T, :T]
        score = score.masked_fill(causal_mask == 0, float("-inf"))
        prob = self.drop(F.softmax(score, dim=-1))
        output = (prob @ V_new).transpose(1, 2).reshape(B, T, self.config["d_model"])
        output = self.ffn(output)

        for b, length in enumerate(lengths):
            # 只复制真实token的KV到该请求最终使用的缓存。
            k = K_new[b : b + 1, :, :length, :]
            v = V_new[b : b + 1, :, :length, :]
            caches[b].write(layer_idx, 0, k, v)
        return output

    def decode_batch(self, x, positions, caches, layer_idx):
        B, T, _ = x.shape
        if T != 1 or positions.shape != (B, 1) or len(caches) != B:
            raise ValueError("decode_batch 要求每个请求恰有一个 token/cache")

        Q = self.mutihead(self.Wq(x))  # [B,H,1,D]
        K_new = self.mutihead(self.Wk(x))
        V_new = self.mutihead(self.Wv(x))

        Q = self.RoPE(Q, positions)
        K_new = self.RoPE(K_new, positions)

        # 每个请求持有一个 KVCache，本层写入同一个逻辑位置。
        for b, cache in enumerate(caches):
            cache.write(
                layer_idx, cache.length, K_new[b : b + 1], V_new[b : b + 1]
            )

        # 公共 length 尚未提交，读取范围需包括本轮新写入的 token。
        lengths = [cache.length + T for cache in caches]
        Lmax = max(lengths)

        H, D = Q.size(1), Q.size(-1)

        K = Q.new_zeros(
            B, H, Lmax, D
        )  # 创建一个形状为 (B, H, Lmax, D) 的全零张量，同时自动沿用 Q 的 dtype 和 device（和其他张量属性）。
        V = Q.new_zeros(B, H, Lmax, D)

        for b, cache in enumerate(caches):
            n = lengths[b]
            k, v = cache.get_layer(layer_idx, n)
            K[b, :, :n] = k[0]
            V[b, :, :n] = v[0]

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

    def prefill_batch(self, x, positions, lengths, caches, layer_idx):
        attn_output = self.atte.prefill_batch(
            self.layerNorm1(x), positions, lengths, caches, layer_idx
        )
        x = x + self.drop(attn_output)
        x = x + self.drop(self.ffn(self.layerNorm2(x)))
        return x

    def decode_batch(self, x, positions, caches, layer_idx):
        attn_output = self.atte.decode_batch(
            self.layerNorm1(x), positions, caches, layer_idx
        )
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

    @torch.no_grad()
    def prefill_batch(self, prompts, pad_token_id=0, max_new_tokens=None):
        """
        prompts: [[token_id, ...], [token_id, ...], ...]多请求
        max_new_tokens: 可选的逐请求生成上限；缺省时缓存容量为 max_seq_len。
        返回:
            last_logits: [B, vocab_size]
            caches: 每个请求一个独立的 KVCache，包含所有层的 K/V
        """
        if self.training:
            raise ValueError("请先调用 model.eval()")
        if not prompts or any(len(ids) == 0 for ids in prompts):
            raise ValueError("每个 prompt 都必须非空")
        if any(len(ids) > self.config["max_seq_len"] for ids in prompts):
            raise ValueError("prompt 超过 max_seq_len")
        if not 0 <= pad_token_id < self.config["vocab_size"]:
            raise ValueError("pad_token_id 必须是有效的 token ID")

        device = self.token_embedding.weight.device
        lengths = [len(ids) for ids in prompts] # batch内不同请求的长度可能不同
        batch_size = len(prompts)
        max_len = max(lengths)
        max_seq_len = self.config["max_seq_len"]

        if max_new_tokens is None:
            capacities = [max_seq_len] * batch_size
        else:
            if isinstance(max_new_tokens, int) and max_new_tokens>=0:
                max_new_tokens = [max_new_tokens] * batch_size
            else:
                raise ValueError(
                    "max_new_tokens应为非负整数或与请求等长的非负整数序列"
                )
            
            capacities = [
                min(max_seq_len, length + n)
                for length, n in zip(lengths, max_new_tokens)
            ]

        input_ids = torch.full(  # 填充至等长batch
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )

        for b, ids in enumerate(prompts):
            input_ids[b, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )

        positions = torch.arange(max_len, device=device)
        x = self.drop(self.token_embedding(input_ids))
        #创建KVCache缓存
        caches = [KVCache(len(self.trans), capacity) for capacity in capacities]
        for layer_idx, block in enumerate(self.trans):
            x = block.prefill_batch(x, positions, lengths, caches, layer_idx)

        rows = torch.arange(batch_size, device=device)
        last_positions = torch.tensor(lengths, dtype=torch.long, device=device) - 1
        # 仅投影各请求最后一个有效位置，避免生成 [B,T,vocab_size] 的大 logits。
        last_logits = self.lm_head(self.layerNorm(x[rows, last_positions]))
        # 所有层都写完后，每个请求只推进一次缓存长度。
        for cache, length in zip(caches, lengths):
            cache.advance(length)

        return last_logits, caches

    @torch.no_grad()
    def decode_batch(
        self,
        input_token_ids: list[int],
        caches: list[KVCache],
    ):
        if not caches or len(input_token_ids) != len(caches):
            raise ValueError("每个输入 token 必须对应一个 KVCache")
        if self.training:
            raise ValueError("批量 decode 请先调用 model.eval()")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("同一批次不能重复使用一个请求的 KVCache")
        for cache in caches:
            if len(cache.keys) != len(self.trans):
                raise ValueError("缓存层数与模型不一致")

            if cache.length >= self.config["max_seq_len"]:
                raise ValueError("超过模型上下文上限")

            if cache.length >= cache.capacity:
                raise ValueError("KV cache 容量不足")

        device = self.token_embedding.weight.device
        positions = torch.tensor(
            [cache.length for cache in caches],
            dtype=torch.long,
            device=device,
        )[:, None]

        input_ids = torch.tensor(
            input_token_ids,
            dtype=torch.long,
            device=device,
        )[:, None]  # [B,1]由于每次只decode一个token

        x = self.drop(self.token_embedding(input_ids))

        for layer_idx, block in enumerate(self.trans):
            x = block.decode_batch(x, positions, caches, layer_idx)
        logits = self.lm_head(self.layerNorm(x))  # [B,1,vocab_size]
        for cache in caches:
            cache.advance(1)
        return logits


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
