from typing import List

import torch

from .guides import Guide

__all__ = ["Guidance"]


class Guidance:
    """
    TODO
    """

    def __init__(
        self,
        guides: List[Guide] = [],
    ) -> None:
        self._guides = guides

    def apply(self, score: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TODO
        """
        guided = score.clone()
        for g in self._guides:
            guided -= self._pad_zeros(guided, g.apply(x, **kwargs))
        return guided

    def __call__(self, score: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TODO
        """
        return self.apply(score, x, **kwargs)

    def add(self, mask: Guide) -> None:
        """
        TODO
        """
        self._guides.append(mask)

    def _pad_zeros(self, orig: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        """
        new_x = torch.zeros_like(orig)
        new_x[:, : x.shape[1]] = x
        return new_x
