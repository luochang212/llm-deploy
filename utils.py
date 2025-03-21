import os
import torch


def is_gpu_available():
    """GPU 是否可用"""
    return torch.cuda.is_available()


def get_free_memory(gpu_id: int) -> int:
    """获取指定GPU的剩余显存"""
    free_memory, _ = torch.cuda.mem_get_info(gpu_id)
    return free_memory


def pick_gpu() -> int:
    """选择显存剩余最多的GPU"""
    max_free = -1
    gpu_id = 0

    for i in range(torch.cuda.device_count()):
        free = get_free_memory(i)
        if free > max_free:
            max_free = free
            gpu_id = i

    return gpu_id, f'{max_free / 1024**2:.2f} MB'


def torch_gc():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
