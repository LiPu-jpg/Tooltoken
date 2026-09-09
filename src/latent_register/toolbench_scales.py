"""Measure compiler scales from materialized embedding rows, including ZeRO-3."""
from contextlib import nullcontext
import math

import torch


def mean_row_norm(weight, *, chunk_rows=2048):
    if weight.is_meta or weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError('Compiler scale requires a nonempty materialized embedding matrix')
    if chunk_rows < 1:
        raise ValueError('chunk_rows must be positive')
    total = 0.0
    for start in range(0, weight.shape[0], chunk_rows):
        total += float(weight[start:start + chunk_rows].float().norm(dim=-1).double().sum())
    result = total / weight.shape[0]
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Compiler scale must be finite and positive; no epsilon fallback')
    return result


@torch.no_grad()
def embedding_scale(weight):
    sharded = hasattr(weight, 'ds_id')
    if sharded:
        # HF can enable ZeRO-3 from the distributed config even when the
        # Accelerate zero3_init_flag is false. Outside this context weight.data
        # may be a one-dimensional empty placeholder, whose norm is zero.
        from deepspeed.zero import GatheredParameters
        context = GatheredParameters([weight], modifier_rank=None)
    else:
        context = nullcontext()
    before = list(weight.shape)
    with context:
        value = mean_row_norm(weight)
        shape = list(weight.shape)
        if sharded and tuple(shape) != tuple(weight.ds_shape):
            raise ValueError('Gathered embedding shape differs from declared full shape')
    return value, {'zero3_gathered': sharded, 'shape_before_gather': before,
                   'materialized_shape': shape, 'mean_row_norm': value}
