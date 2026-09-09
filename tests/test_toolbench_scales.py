import json
import math
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from latent_register.toolbench_checkpoint import sha256
from latent_register.toolbench_repair_scales import repair
from latent_register.toolbench_scales import embedding_scale, mean_row_norm


def test_empty_placeholder_reproduces_old_epsilon_but_new_path_rejects():
    weight = torch.empty(0)
    assert max(float(weight.float().norm(dim=-1).mean()), 1e-6) == 1e-6
    with pytest.raises(ValueError, match='materialized'):
        embedding_scale(weight)


def test_zero3_norm_is_measured_inside_gather_then_partition_is_restored(monkeypatch):
    full = torch.tensor([[3., 4.], [0., 2.], [6., 8.]])
    weight = torch.nn.Parameter(torch.empty(0))
    weight.ds_id, weight.ds_shape = 0, full.shape
    entered = []
    @contextmanager
    def gather(parameters, modifier_rank):
        assert parameters == [weight] and modifier_rank is None
        entered.append(True)
        weight.data = full.clone()
        try:
            yield
        finally:
            weight.data = torch.empty(0)
    # Match the installed public API. Making a fake deepspeed.zero package
    # would hide the import error that previously failed before training.
    monkeypatch.delitem(sys.modules, 'deepspeed.zero', raising=False)
    monkeypatch.setitem(sys.modules, 'deepspeed',
                        SimpleNamespace(zero=SimpleNamespace(GatheredParameters=gather)))
    value, profile = embedding_scale(weight)
    assert value == pytest.approx(17/3)
    assert entered == [True] and weight.shape == (0,)
    assert profile['shape_before_gather'] == [0]
    assert profile['materialized_shape'] == [3, 2] and profile['zero3_gathered']


@pytest.mark.parametrize('weight', [torch.zeros(2, 3), torch.full((2, 3), float('nan')),
    torch.full((2, 3), float('inf')), torch.empty((2, 3), device='meta')])
def test_invalid_norm_never_silently_falls_back(weight):
    with pytest.raises(ValueError): mean_row_norm(weight)


def test_norm_is_independent_of_chunk_size():
    torch.manual_seed(1)
    weight = torch.randn(31, 12).to(torch.bfloat16)
    assert mean_row_norm(weight, chunk_rows=7) == pytest.approx(mean_row_norm(weight, chunk_rows=31), abs=1e-10)


def make_parent(tmp_path, bad_scale=True):
    parent = tmp_path/'parent'; (parent/'backbone').mkdir(parents=True)
    save_file({'model.embed_tokens.weight': torch.tensor([[3., 4.], [0., 2.]]),
               'lm_head.weight': torch.tensor([[6., 8.], [3., 4.]])}, parent/'backbone/model.safetensors')
    norm = math.log(1e-6 if bad_scale else 2.0)
    runtime = {'compilers': {'output_compiler.generator.log_output_norm': torch.tensor(norm, dtype=torch.bfloat16),
        'memory_compiler.log_output_norm': torch.full((8,), norm, dtype=torch.bfloat16),
        'other.weight': torch.randn(3,4)}, 'backbone_buffers': {'rotary_emb.inv_freq': torch.tensor([1., 0.01])}}
    torch.save(runtime, parent/'runtime.pt')
    (parent/'agent.json').write_text(json.dumps({'metadata': {'updates': 172, 'train_api_identities': ['train-api']}}))
    manifest={str(p.relative_to(parent)):sha256(p) for p in parent.rglob('*') if p.is_file()}
    (parent/'SHA256.json').write_text(json.dumps(manifest))
    return parent, manifest


def test_repair_preserves_all_backbone_and_other_compiler_bytes(tmp_path):
    parent, manifest = make_parent(tmp_path)
    output=tmp_path/'derived'
    report = repair(parent, output)
    assert report['optimizer_updates'] == report['api_documents_used'] == 0
    changes = report['changes']
    assert changes['output_compiler.generator.log_output_norm']['measured_mean_row_norm'] == 7.5
    assert changes['memory_compiler.log_output_norm']['measured_mean_row_norm'] == 3.5
    for name, digest in manifest.items(): assert sha256(parent/name) == digest
    assert (output/'backbone/model.safetensors').stat().st_ino == (parent/'backbone/model.safetensors').stat().st_ino
    assert (output/'runtime.pt').stat().st_ino != (parent/'runtime.pt').stat().st_ino
    config = json.loads((output/'agent.json').read_text())
    assert config['metadata']['updates'] == 172
    assert config['metadata']['train_api_identities'] == ['train-api']
    assert config['metadata']['initialization_repair']['not_an_unmodified_checkpoint_result']
    for name,digest in json.loads((output/'SHA256.json').read_text()).items(): assert sha256(output/name)==digest
    with pytest.raises(FileExistsError): repair(parent,output)


def test_repair_refuses_retuning_a_normal_scale(tmp_path):
    parent,_=make_parent(tmp_path,bad_scale=False)
    with pytest.raises(ValueError,match='not the diagnosed'): repair(parent,tmp_path/'derived')
    assert not (tmp_path/'derived').exists()
