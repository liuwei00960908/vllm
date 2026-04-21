from dataclasses import dataclass
import torch


@dataclass
class BlockContainer:
    block_ids: list[int]
    size: int

@dataclass
class Clusters:
    cluster_blocks: list[BlockContainer]
    cluster_centers: torch.Tensor
    mean: torch.Tensor