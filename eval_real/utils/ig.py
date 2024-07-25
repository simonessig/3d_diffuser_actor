import torch


class InteractiveGuidance:
    def __init__(self, device="cuda"):
        self.device = device

    def apply(self, score: torch.tensor, x: torch.tensor) -> torch.tensor:
        new_score = score.clone().to(self.device)
        mask = torch.zeros(score.shape, dtype=torch.float32, device=self.device)
        max_score = torch.max(score)
        if x[..., 1] < 0.05:
            new_score[..., 0] = 0
            new_score[..., 1] = -max_score
            new_score[..., 2] = 0
            # print(mask)

        return new_score

        mask = torch.zeros(score.shape, dtype=torch.float32, device=self.device)
        if x[..., 1] > 0:
            mask[..., 1] = 5
            # print(mask)

        return score.clone().to(self.device) + mask
