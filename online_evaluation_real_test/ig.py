import torch


class InteractiveGuidance:
    def __init__(self, device="cuda"):
        self.device = device

    def apply(self, score: torch.tensor, x: torch.tensor) -> torch.tensor:
        # return score
        mask = torch.zeros(score.shape, dtype=torch.float32, device=self.device)
        if x[..., 1] > 0:
            mask[..., 1] = 5
            # print(mask)

        return score.clone().to(self.device) + mask
