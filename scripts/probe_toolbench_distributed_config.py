"""CPU-only check of the installed DeepSpeed communicator/config contract."""
import argparse
import json
from pathlib import Path

import deepspeed
import torch.distributed as dist
from deepspeed.runtime.config import DeepSpeedConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    deepspeed.init_distributed(dist_backend='gloo', auto_mpi_discovery=False)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == deepspeed.comm.get_world_size() == 4
        assert rank == deepspeed.comm.get_rank()
        config = DeepSpeedConfig({'train_batch_size': 4,
            'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1})
        assert config.world_size == 4
        ranks = [None] * world
        dist.all_gather_object(ranks, rank)
        assert ranks == list(range(world))
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        report = {'rank': rank, 'torch_world_size': world,
                  'deepspeed_world_size': deepspeed.comm.get_world_size(),
                  'config_world_size': config.world_size, 'gathered_ranks': ranks,
                  'deepspeed_version': deepspeed.__version__,
                  'passed': True, 'gpu_used': False, 'model_loaded': False}
        (args.output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
