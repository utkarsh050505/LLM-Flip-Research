import os

def setup_ddp_environment():
    try:
        import torch
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    except ImportError:
        pass


def is_main_process() -> bool:
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except ImportError:
        return True


def cleanup_ddp():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


def get_world_size_and_rank():
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return 1, 0
        return dist.get_world_size(), dist.get_rank()
    except ImportError:
        return 1, 0


def gather_data_ddp(data):
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return data
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, data)
        flat = []
        for g in gathered:
            if isinstance(g, list):
                flat.extend(g)
            elif g is not None:
                flat.append(g)
        return flat
    except ImportError:
        return data


def split_list_among_ranks(data_list):
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return data_list
        world_size, rank = get_world_size_and_rank()
        return data_list[rank::world_size]
    except ImportError:
        return data_list
