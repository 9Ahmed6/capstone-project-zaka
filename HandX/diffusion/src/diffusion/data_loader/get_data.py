from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from .. import dist as dist_utils
from ..config import DataLoaderConfig

def get_dataloader(dataset:Dataset, cfg:DataLoaderConfig):
    is_distributed = dist_utils.is_dist_avail_and_initialized()
    sampler = DistributedSampler(dataset, shuffle=cfg.shuffle) if is_distributed else None
    if sampler is not None:
        shuffle = False
    else:
        shuffle = cfg.shuffle

    if cfg.batch_size % dist_utils.get_world_size() != 0:
        raise ValueError(f"Batch size {cfg.batch_size} must be divisible by world size {dist_utils.get_world_size()}.")

    batch_size = cfg.batch_size // dist_utils.get_world_size()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=dataset.dataset.collate_fn if isinstance(dataset, Subset) else dataset.collate_fn,
    )

    return loader
