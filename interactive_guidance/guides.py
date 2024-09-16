from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch

__all__ = ["StaticGuide", "CamGuide", "PointGuide"]


class Guide(ABC):
    """
    TODO
    """

    def __init__(
        self,
        mult: float = 1,
        condition: Optional[Callable[[], bool]] = None,
        device: str = "cpu",
    ) -> None:
        self._mult = mult
        self._device = device
        self._condition = condition

    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TODO
        x (n,3)

        out (n,3)
        """
        if self._condition is not None and not self._condition(**kwargs):
            return torch.zeros_like(x, device=x.device)

        return self._mult * self._get_score(x.clone().to(self._device)).to(x.device)

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        TODO
        x (n,3)

        out (n,3)
        """
        return self.apply(x, **kwargs)

    @abstractmethod
    def _get_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        x (n,3)

        out (n,3)
        """
        return torch.zeros_like(x)


class StaticGuide(Guide):
    """
    TODO
    """

    def __init__(
        self,
        callback: Callable[[torch.Tensor], torch.Tensor],
        mult: float = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__(mult, None, device)
        self._callback = callback

    def _get_score(self, x: torch.Tensor) -> torch.Tensor:
        return self._callback(x)


class CamGuide(Guide):
    """
    TODO
    """

    def __init__(
        self,
        mask: torch.Tensor,
        intrinsics: torch.Tensor,
        pos: torch.Tensor,
        rot_mat: torch.Tensor,
        mask_only_frame: bool = False,
        mult: float = 1,
        condition: Optional[Callable[[], bool]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(mult, condition, device)

        self._mask = mask.to(self._device)

        intrinsics = torch.tensor(intrinsics, dtype=torch.float32).to(self._device)

        extrinsics = torch.eye((4), dtype=torch.float32).to(self._device)
        extrinsics[:3, :3] = torch.tensor(rot_mat, dtype=torch.float32)

        self._inv_rot_mat = extrinsics.clone().T.to(self._device)

        # Rotate position, because translation is done after rotation is extrinsics matrix
        h_pos = torch.tensor([*pos, 1], dtype=torch.float32).to(self._device)
        h_pos = extrinsics.matmul(h_pos)
        extrinsics[:3, 3] = -h_pos[:3]

        self._world_to_pixel_mat = intrinsics.matmul(extrinsics).to(self._device)

        self._width = mask.shape[0]
        self._height = mask.shape[1]
        self._mask_only_frame = mask_only_frame

    def _get_score(self, x: torch.Tensor) -> torch.Tensor:
        return self._coord_to_mask(x[:, :3])

    def _coord_to_mask(self, coord: torch.Tensor) -> torch.Tensor:
        """
        TODO
        coord (n, 3)
        out (n, 3)
        """
        i, j, d = self._coord_to_pix(coord)

        masked = self._mask[i.clamp(0, self._width - 1), self._height - j.clamp(1, self._height)]

        h_masked = torch.hstack((masked, torch.ones((masked.shape[0], 1), device=masked.device))).T
        masked = self._inv_rot_mat.matmul(h_masked)[:3].T

        # Exclude points behind camera
        cond = d > 0

        # Exclude points out of frame
        if self._mask_only_frame:
            inside_frame = torch.logical_and(
                torch.logical_and(i >= 0, i < self._width),
                torch.logical_and(j >= 0, j < self._height),
            )
            cond = torch.logical_and(cond, inside_frame)

        cond = torch.stack((cond, cond, cond), dim=0).T
        zeros = torch.zeros(masked.shape, dtype=torch.float32, device=coord.device)
        return torch.where(cond, masked, zeros)

    def _coord_to_pix(self, coord: torch.Tensor) -> torch.Tensor:
        """
        Projects a world point to pixel coordinates
        coord (n, 3)
        out (n, 3)
        """
        # Homogenous coordinates
        h_coord = torch.hstack((coord, torch.ones((coord.shape[0], 1), device=coord.device))).T

        world_pix_tran = self._world_to_pixel_mat.matmul(h_coord)
        w = world_pix_tran[2, :]
        projected = world_pix_tran / w

        i, j = torch.round(projected[0:2]).type(torch.int)

        return i, j, w


class PointGuide(Guide):
    """
    TODO
    """

    def __init__(
        self,
        point: torch.Tensor,
        radius: float = 1,
        mask_outside: bool = False,
        outward: bool = False,
        mult: float = 1,
        condition: Optional[Callable[[], bool]] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(mult, condition, device)

        self._point = point
        self._radius = radius
        self._mask_outside = mask_outside
        self._outward = outward

    def _get_score(self, x: torch.Tensor) -> torch.Tensor:
        scores = self._point - x[:, :3]
        norms = torch.linalg.norm(scores, dim=1)
        scores = torch.nn.functional.normalize(scores)

        if self._outward:
            scores = -scores

        mask = norms < self._radius if self._mask_outside else norms > self._radius
        scores[mask] = 0

        return scores
