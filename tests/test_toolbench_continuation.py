import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from test_toolbench_agent import agent, tools, row
from latent_register.toolbench_checkpoint import save_agent, load_agent, sha256
from latent_register.toolbench_data import FINISH, compact
from latent_register.toolbench_continuation import inspect_continuation, load_trainable_agent, RuntimeTrainingState


@pytest.mark.parametrize('mode', ['full', 'compiler_only'])
def test_serving_export_continues_real_schema_gradients(agent, tools, tmp_path, mode):
    save_agent(agent, tmp_path/'parent', metadata={})
    model = load_trainable_agent(tmp_path/'parent', train_mode=mode, gradient_checkpointing=True)
    for before, after in zip(agent.parameters(), model.parameters()):
        assert torch.equal(before, after)
    _, memory = model.compile([tools['weather/forecast']])
    model.schema_loss(tools['weather/forecast'], memory[0], tasks_per_step=5).backward()
    assert model.memory_compiler.queries.grad.abs().sum() > 0
    embedding = model.backbone.get_input_embeddings().weight
    assert (embedding.grad is not None) == (mode == 'full')
    if mode == 'full':
        assert embedding.grad.abs().sum() > 0
        assert model.backbone.lm_head.weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in model.output_compiler.parameters())


def test_parent_contract_rejects_new_split_architecture_and_false_resume(agent, tmp_path):
    audit={'source_sha256':{'tools':'a','trajectories':'b'},'train_api_identities':['seen-api']}
    metadata={**audit,'train_mode':'full','schema_weight':0.5,'updates':172,
              'world_size':4,'effective_full_batch':8}
    save_agent(agent,tmp_path/'parent',metadata=metadata)
    args=SimpleNamespace(train_mode='full',schema_weight=0.5,condition='memory',compiler_rank=8,
        memory_slots=8,memory_kind='structured',memory_width=32,memory_depth=2,memory_heads=4,
        max_context_length=2048,max_document_length=512,max_target_length=512,max_updates=500)
    contract=inspect_continuation(tmp_path/'parent',args,audit)
    assert contract['parent_updates']==172 and contract['optimizer_restored'] is False
    changed=copy.deepcopy(audit);changed['train_api_identities']=['different-api']
    with pytest.raises(ValueError,match='training contract'):inspect_continuation(tmp_path/'parent',args,changed)
    args.memory_slots=16
    with pytest.raises(ValueError,match='model interface'):inspect_continuation(tmp_path/'parent',args,audit)
    args.memory_slots=8;args.max_updates=172
    with pytest.raises(ValueError,match='cumulative target'):inspect_continuation(tmp_path/'parent',args,audit)


def test_accelerator_restores_optimizer_scheduler_rng_and_rope(agent, tools, tmp_path):
    accelerator=Accelerator(cpu=True)
    optimizer=torch.optim.AdamW(agent.parameters(),lr=1e-3)
    model, optimizer=accelerator.prepare(agent,optimizer)
    scheduler=get_cosine_schedule_with_warmup(optimizer,0,10)
    runtime=RuntimeTrainingState(accelerator.unwrap_model(model),{'synthetic':'same-contract'})
    runtime.cursor={'cumulative_updates':1,'next_microbatch':1,'epoch':0}
    accelerator.register_for_checkpointing(scheduler,runtime)
    def update():
        optimizer.zero_grad()
        _,memory=model.compile([tools['weather/forecast']])
        loss=model.schema_loss(tools['weather/forecast'],memory[0],tasks_per_step=1)
        accelerator.backward(loss);optimizer.step();scheduler.step()
        return loss.detach().clone()
    update()
    accelerator.save_state(str(tmp_path/'state'))
    expected_rng=torch.rand(5)
    expected_loss=update()
    expected_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    # Nonpersistent RoPE corruption is deliberately not recoverable from ordinary state_dict.
    rope=model.backbone.model.rotary_emb
    with torch.no_grad():rope.inv_freq.add_(0.25)
    runtime.cursor={}
    accelerator.load_state(str(tmp_path/'state'))
    assert torch.equal(torch.rand(5),expected_rng)
    assert runtime.cursor['cumulative_updates']==1
    actual_loss=update()
    assert torch.equal(actual_loss,expected_loss)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,expected_weights[key],rtol=0,atol=0)


def test_real_cli_continues_to_cumulative_target_and_saves_full_state(agent, tools, row, tmp_path):
    registry=tmp_path/'tools.jsonl'
    registry.write_text(''.join(compact({'api_identity':key,'document':tool.document,
        'parameters':tool.parameters,'aliases':tool.aliases,'split':'train'})+'\n'
        for key,tool in tools.items() if key != FINISH))
    trajectories=tmp_path/'train.jsonl';trajectories.write_text(compact(row)+'\n')
    parent=tmp_path/'parent'
    save_agent(agent,parent,metadata={'source_sha256':{'tools':sha256(registry),'trajectories':sha256(trajectories)},
        'train_api_identities':sorted(set(tools)-{FINISH}),'train_mode':'full','schema_weight':0.5,
        'updates':1,'world_size':1,'effective_full_batch':2})
    output=tmp_path/'continued'
    command=[sys.executable,'-m','latent_register.train_toolbench_agent','--tools',str(registry),
        '--trajectories',str(trajectories),'--source-format','toolbench','--output-dir',str(output),
        '--continue-from-export',str(parent),'--epochs','1','--max-updates','2',
        '--gradient-accumulation-steps','2','--compiler-rank','8','--candidate-count','3',
        '--memory-width','32','--memory-heads','4','--max-context-length','2048',
        '--max-document-length','512','--max-target-length','512','--save-training-state-every','1']
    env={**os.environ,'ACCELERATE_USE_CPU':'true','OMP_NUM_THREADS':'1',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads((output/'TRAINING_COMPLETE.json').read_text())['updates']==2
    ready=json.loads((output/'training-state/update-2/READY.json').read_text())
    assert ready['cursor']['new_updates']==1 and ready['cursor']['next_microbatch']==2
    assert any('optimizer' in name for name in ready['files'])
    assert any('random_states' in name for name in ready['files'])
    for name,digest in ready['files'].items():assert sha256(output/'training-state/update-2'/name)==digest
    continued=load_agent(output/'checkpoint')
    assert not torch.equal(continued.backbone.get_input_embeddings().weight,agent.backbone.get_input_embeddings().weight)
