import torch


class EMINN(torch.nn.Module):
    """Finite-agent neural approximation of marginal value W = dV/da."""

    def __init__(self, input_dim: int, width: int, layers: int):
        super().__init__()
        blocks = [torch.nn.Linear(input_dim, width), torch.nn.Tanh()]
        for _ in range(1, layers):
            blocks.extend([torch.nn.Linear(width, width), torch.nn.Tanh()])
        blocks.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*blocks)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, agg], dim=1))
