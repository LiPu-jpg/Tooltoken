"""Create an explicit derived export repairing the confirmed epsilon-scale defect.

No optimizer, query, document, simulator, test answer or GPU is used. Backbone
bytes stay identical. The two scale tensors are reset to measured embedding
norms; all other compiler parameters and runtime buffers remain unchanged.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open

from .toolbench_checkpoint import sha256
from .toolbench_scales import mean_row_norm


def stored_embedding_norm(backbone, name, chunk_rows=2048):
    index = backbone / 'model.safetensors.index.json'
    filename = json.loads(index.read_text())['weight_map'][name] if index.exists() else 'model.safetensors'
    with safe_open(backbone / filename, framework='pt', device='cpu') as handle:
        view = handle.get_slice(name)
        shape = view.get_shape()
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError('Expected a complete embedding matrix')
        total = 0.0
        for start in range(0, shape[0], chunk_rows):
            chunk = view[start:start + chunk_rows, :]
            total += mean_row_norm(chunk) * chunk.shape[0]
        return total / shape[0], shape


def repair(parent, output):
    if output.exists():
        raise FileExistsError('Never overwrite an export or previous repair')
    manifest = json.loads((parent / 'SHA256.json').read_text())
    for name, expected in manifest.items():
        item = parent / name
        if not item.resolve().is_relative_to(parent.resolve()) or sha256(item) != expected:
            raise ValueError(f'Parent export authentication failed: {name}')
    runtime = torch.load(parent / 'runtime.pt', map_location='cpu', weights_only=True)
    changes = {}
    for key, embedding in [('output_compiler.generator.log_output_norm', 'lm_head.weight'),
                            ('memory_compiler.log_output_norm', 'model.embed_tokens.weight')]:
        old = runtime['compilers'][key]
        # This command fixes only the observed epsilon-initialization defect.
        # It cannot silently retune otherwise learned scales.
        if not torch.allclose(old.float().exp(), torch.full_like(old.float(), 1e-6), rtol=0.02, atol=0):
            raise ValueError(f'{key} is not the diagnosed epsilon-scale checkpoint')
        norm, shape = stored_embedding_norm(parent / 'backbone', embedding)
        new = torch.full_like(old, math.log(norm))
        changes[key] = {'old_log_values': old.float().tolist(), 'new_log_values': new.float().tolist(),
                        'measured_mean_row_norm': norm, 'embedding': embedding, 'full_shape': shape,
                        'actual_stored_scale': new.float().exp().tolist()}
        runtime['compilers'][key] = new
    config = json.loads((parent / 'agent.json').read_text())
    if config['metadata'].get('initialization_repair'):
        raise ValueError('Do not apply this repair twice')
    report = {'kind': 'zero3_epsilon_scale_repair_v1',
        'parent_checkpoint': str(parent.resolve()), 'parent_manifest_sha256': sha256(parent / 'SHA256.json'),
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'changes': changes,
        'optimizer_updates': 0, 'training_examples_used': 0, 'api_documents_used': 0,
        'backbone_weights_modified': False, 'other_compiler_weights_modified': False,
        'previous_updates_not_reclassified_as_valid_joint_training': config['metadata']['updates'],
        'not_an_unmodified_checkpoint_result': True}
    config['metadata']['initialization_repair'] = report
    output.mkdir(parents=True, exist_ok=False)
    # Immutable shared backbone files consume no extra 16 GB copy. Never open
    # linked files for writing; all modified files are created separately.
    changed_files = {'runtime.pt', 'agent.json'}
    for name in manifest:
        if name in changed_files:
            continue
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(parent / name, target)
    torch.save(runtime, output / 'runtime.pt')
    (output / 'agent.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    (output / 'INITIALIZATION_REPAIR.json').write_text(json.dumps(report, indent=2) + '\n')
    new_manifest = {name: digest for name, digest in manifest.items() if name not in changed_files}
    for name in [*changed_files, 'INITIALIZATION_REPAIR.json']:
        new_manifest[name] = sha256(output / name)
    (output / 'SHA256.json').write_text(json.dumps(new_manifest, sort_keys=True, indent=2) + '\n')
    original = torch.load(parent / 'runtime.pt', map_location='cpu', weights_only=True)
    for group in original:
        for name, value in original[group].items():
            if group == 'compilers' and name in changes:
                continue
            if not torch.equal(value, runtime[group][name]):
                raise AssertionError(f'Unexpected parameter or buffer change: {group}/{name}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    result = repair(args.parent, args.output)
    print(json.dumps({'repaired': True, 'changes': result['changes'], 'optimizer_updates': 0}))


if __name__ == '__main__':
    main()
