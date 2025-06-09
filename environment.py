# 환경 정의를 담당하는 모듈
# -*- coding: utf-8 -*-
"""
환경 객체는 에이전트와 장애물, 보상 위치를 관리한다.
"""
from dataclasses import dataclass
import random
from typing import List, Tuple

@dataclass
class Obstacle:
    x: int
    y: int
    w: int
    h: int

@dataclass
class Reward:
    x: int
    y: int
    r: int = 5

class Environment:
    """2D 격자 환경"""
    def __init__(self, width: int = 400, height: int = 300):
        self.width = width
        self.height = height
        # 고정 장애물 설정
        self.obstacles: List[Obstacle] = [
            Obstacle(100, 50, 40, 40),
            Obstacle(200, 150, 60, 20),
            Obstacle(300, 80, 30, 70),
        ]
        self.reward = Reward(50, 200)

    def random_position(self) -> Tuple[int, int]:
        """환경 범위 내 무작위 위치 반환"""
        return random.randint(0, self.width-1), random.randint(0, self.height-1)

    def reset_reward(self) -> None:
        """보상 위치를 새로운 무작위 지점으로 이동"""
        rx, ry = self.random_position()
        self.reward.x, self.reward.y = rx, ry

    def check_collision(self, x: int, y: int, r: int) -> bool:
        """원형 객체와 장애물 간 충돌 여부 확인"""
        for obs in self.obstacles:
            if (obs.x - r <= x <= obs.x + obs.w + r and
                obs.y - r <= y <= obs.y + obs.h + r):
                return True
        return False
