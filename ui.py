# UI 모듈
# -*- coding: utf-8 -*-
"""tkinter 기반 2D 시각화"""
import tkinter as tk
from typing import List
import torch

from environment import Environment, Obstacle
from agent import Agent
from network import SimpleTransformer

class App:
    FPS = 30
    def __init__(self):
        self.env = Environment()
        self.agent = Agent(20, 20)
        self.net = SimpleTransformer(num_obstacles=len(self.env.obstacles))
        self.root = tk.Tk()
        self.root.title("AI Agent")
        self.canvas = tk.Canvas(self.root, width=self.env.width, height=self.env.height, bg="white")
        self.canvas.pack()
        self.draw_static_elements()
        self.agent_item = self.canvas.create_oval(0, 0, self.agent.r*2, self.agent.r*2, fill="blue")
        self.reward_item = self.canvas.create_oval(0, 0, self.env.reward.r*2, self.env.reward.r*2, fill="green")
        self.update_reward()
        self.root.after(int(1000/self.FPS), self.update_loop)

    def draw_static_elements(self) -> None:
        """장애물 등 변하지 않는 요소 그리기"""
        for obs in self.env.obstacles:
            self.canvas.create_rectangle(obs.x, obs.y, obs.x + obs.w, obs.y + obs.h, fill="grey")

    def update_reward(self) -> None:
        r = self.env.reward
        self.canvas.coords(self.reward_item, r.x - r.r, r.y - r.r, r.x + r.r, r.y + r.r)

    def compute_action(self) -> List[int]:
        """신경망으로부터 이동 방향을 결정"""
        agent_pos = (self.agent.x / self.env.width, self.agent.y / self.env.height)
        reward_pos = (self.env.reward.x / self.env.width, self.env.reward.y / self.env.height)
        obs_pos = [
            (obs.x / self.env.width, obs.y / self.env.height) for obs in self.env.obstacles
        ]
        with torch.no_grad():
            logits = self.net(agent_pos, reward_pos, obs_pos)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
        return action

    def apply_action(self, action: int) -> None:
        dx, dy = 0, 0
        if action == 0:  # up
            dy = -self.agent.speed
        elif action == 1:  # down
            dy = self.agent.speed
        elif action == 2:  # left
            dx = -self.agent.speed
        elif action == 3:  # right
            dx = self.agent.speed
        new_x = self.agent.x + dx
        new_y = self.agent.y + dy
        if not self.env.check_collision(new_x, new_y, self.agent.r):
            self.agent.move(dx, dy, self.env.width, self.env.height)

    def update_loop(self) -> None:
        """프레임 업데이트"""
        action = self.compute_action()
        self.apply_action(action)
        self.canvas.coords(
            self.agent_item,
            self.agent.x - self.agent.r,
            self.agent.y - self.agent.r,
            self.agent.x + self.agent.r,
            self.agent.y + self.agent.r,
        )
        # 보상 획득 여부 확인
        if ((self.agent.x - self.env.reward.x) ** 2 + (self.agent.y - self.env.reward.y) ** 2) <= (self.agent.r + self.env.reward.r) ** 2:
            self.env.reset_reward()
            self.update_reward()
        self.root.after(int(1000/self.FPS), self.update_loop)

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    app = App()
    app.run()
