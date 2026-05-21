import os
import torch
import torch.distributed as dist
import datetime

def is_dist_avail_and_initialized() :
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def broadcast_tensor(tensor:torch.Tensor):
    if is_dist_avail_and_initialized():
        dist.broadcast(tensor, src=0)

def init_distributed_mode(backend='nccl'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print("Distributed training: rank %d, world_size %d" % (rank, world_size))
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f'cuda:{local_rank}'),
            timeout=datetime.timedelta(minutes=10)
        )
        dist.barrier()
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        local_rank = 0


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor:torch.Tensor):
    if not is_dist_avail_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

def gather_tensors(tensor:torch.Tensor):
    if is_main_process():
        gather_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.gather(tensor=tensor, gather_list=gather_list, dst=0)
        return gather_list
    else:
        dist.gather(tensor=tensor, gather_list=None, dst=0)
        return None