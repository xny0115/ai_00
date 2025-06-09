# 신경망 모듈
# -*- coding: utf-8 -*-
"""간단한 2층 Transformer 네트워크 구현"""
from typing import List, Tuple
import torch
from torch import nn
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMultiheadAttention(nn.Module):
    """Flash-attn이나 SDP에 의존하지 않는 간단한 MultiheadAttention"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, T, _ = query.size()
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """간단한 Transformer 블록"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int | None = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.attn = SimpleMultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x, x, x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, num_obstacles: int, embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.device = DEVICE
        self.num_tokens = 2 + num_obstacles  # 에이전트 + 보상 + 장애물들
        self.embed = nn.Linear(2, embed_dim)
        self.blocks = nn.Sequential(
            TransformerBlock(embed_dim, num_heads),
            TransformerBlock(embed_dim, num_heads),
        )
        self.fc = nn.Linear(embed_dim, 4)  # 상하좌우 확률
        self.to(self.device)

    def forward(self, agent_pos: Tuple[float, float], reward_pos: Tuple[float, float], obstacles: List[Tuple[float, float]]):
        """포지션 정보를 입력 받아 이동 방향 확률 출력"""
        tokens = [agent_pos, reward_pos]
        tokens.extend(obstacles)
        x = torch.tensor(tokens, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, seq, 2)
        emb = self.embed(x)
        enc = self.blocks(emb)
        out = self.fc(enc[:, 0, :])  # 첫 토큰(에이전트)에 대한 출력 사용
        return out
