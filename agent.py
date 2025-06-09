# 에이전트 로직 모듈
# -*- coding: utf-8 -*-
"""에이전트 상태와 이동을 관리"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Agent:
    x: int
    y: int
    r: int = 5
    speed: int = 5

    def position(self) -> Tuple[int, int]:
        return self.x, self.y

    def move(self, dx: int, dy: int, width: int, height: int) -> None:
        """에이전트를 주어진 방향으로 이동"""
        nx = max(0, min(width, self.x + dx))
        ny = max(0, min(height, self.y + dy))
        self.x, self.y = nx, ny
