"""Four-rank real ZeRO-3 scale gathering; no API data or optimizer updates."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist

from latent_register.toolbench_scales import embedding_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-world-size', type=int, default=4)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # DeepSpeedConfig reads DeepSpeed's communicator wrapper. Initializing
    # torch.distributed alone leaves that wrapper unset and reports world=1.
    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == args.expected_world_size
        assert deepspeed.comm.get_world_size() == world
        assert deepspeed.comm.get_rank() == rank
        assert torch.cuda.is_bf16_supported()
        assert torch.cuda.get_device_properties(local_rank).total_memory >= 44 * 1024**3
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier(device_ids=[local_rank])
        (args.output / f'bootstrap-rank-{rank}.json').write_text(json.dumps({
            'rank': rank, 'torch_world_size': world,
            'deepspeed_world_size': deepspeed.comm.get_world_size(),
            'gpu_name': torch.cuda.get_device_name(local_rank),
            'gpu_memory_bytes': torch.cuda.get_device_properties(local_rank).total_memory,
            'optimizer_updates': 0}, indent=2) + '\n')
        config = {'train_batch_size': world, 'train_micro_batch_size_per_gpu': 1,
                  'gradient_accumulation_steps': 1, 'bf16': {'enabled': True},
                  'zero_optimization': {'stage': 3,
                      'offload_param': {'device': 'cpu', 'pin_memory': True}}}
        # This is the installed DeepSpeed API, without a mocked module alias.
        with deepspeed.zero.Init(config_dict_or_path=config, remote_device='cpu',
                                 pin_memory=True, dtype=torch.bfloat16):
            module = torch.nn.Embedding(13, 7, dtype=torch.bfloat16)
        weight = module.weight
        assert hasattr(weight, 'ds_id') and tuple(weight.shape) == (0,)
        with deepspeed.zero.GatheredParameters([weight], modifier_rank=0):
            if rank == 0:
                with torch.no_grad():
                    weight.fill_(1.0)
        before_shape = list(weight.shape)
        value, profile = embedding_scale(weight)
        assert before_shape == [0] and list(weight.shape) == [0]
        assert profile['zero3_gathered'] and profile['materialized_shape'] == [13, 7]
        assert math.isclose(value, math.sqrt(7), rel_tol=0.0, abs_tol=1e-6)
        with deepspeed.zero.GatheredParameters([weight], modifier_rank=None):
            assert torch.equal(weight, torch.ones_like(weight))
        report = {'rank': rank, 'world_size': world, 'passed': True,
                  'gpu_name': torch.cuda.get_device_name(local_rank),
                  'gpu_memory_bytes': torch.cuda.get_device_properties(local_rank).total_memory,
                  'deepspeed_version': deepspeed.__version__,
                  'gathered_parameters_source': inspect.getfile(deepspeed.zero.GatheredParameters),
                  'actual_sharded_parameter': True, 'partition_restored': True,
                  'parameter_values_unchanged': True, 'profile': profile,
                  'optimizer_updates': 0, 'api_documents_read': 0}
        (args.output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports = [None] * world
        dist.all_gather_object(reports, report)
        if rank == 0:
            assert [r['rank'] for r in reports] == list(range(world))
            assert all(r['passed'] for r in reports)
            (args.output / 'VERIFIED.json').write_text(json.dumps(
                {'passed': True, 'world_size': world, 'ranks': reports,
                 'scope': 'real tiny ZeRO-3 gather only; not 8B training certification'}, indent=2) + '\n')
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
