# 신경망 모듈
# -*- coding: utf-8 -*-
"""간단한 2층 Transformer 네트워크 구현"""
from typing import List, Tuple
import torch
from torch import nn

class SimpleTransformer(nn.Module):
    def __init__(self, num_obstacles: int, embed_dim: int = 16, num_heads: int = 2):
        super().__init__()
        self.num_tokens = 2 + num_obstacles  # 에이전트 + 보상 + 장애물들
        self.embed = nn.Linear(2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, 4)  # 상하좌우 확률

    def forward(self, agent_pos: Tuple[float, float], reward_pos: Tuple[float, float], obstacles: List[Tuple[float, float]]):
        """포지션 정보를 입력 받아 이동 방향 확률 출력"""
        tokens = [agent_pos, reward_pos]
        tokens.extend(obstacles)
        x = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)  # (1, seq, 2)
        emb = self.embed(x)
        enc = self.encoder(emb)
        out = self.fc(enc[:, 0, :])  # 첫 토큰(에이전트)에 대한 출력 사용
        return out
