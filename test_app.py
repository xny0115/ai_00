import torch
from network import SimpleTransformer

def test_network_forward():
    net = SimpleTransformer(num_obstacles=3)
    agent = (0.1, 0.1)
    reward = (0.5, 0.5)
    obstacles = [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4)]
    out = net(agent, reward, obstacles)
    assert out.shape == (1, 4)
    prob = torch.softmax(out, dim=-1)
    assert torch.isclose(prob.sum(), torch.tensor(1.0), atol=1e-5)
