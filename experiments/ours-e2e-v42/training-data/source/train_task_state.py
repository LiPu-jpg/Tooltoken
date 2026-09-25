"""Bounded v42 warm-start pilot; explicit NEW optimizer/schedule, not v41 resume."""
import argparse,json,math,time,hashlib,os
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,set_seed
from latent_register.toolbench_checkpoint import load_agent,save_agent,sha256
from latent_register.toolbench_task_state import TaskStateAgent,INTERFACE,validate_state,observed_history
from latent_register.toolbench_hybrid import INTERFACE as PARENT_INTERFACE
from latent_register.toolbench_lora import train_lora_and_compilers,optimizer_groups
from latent_register.toolbench_data import load_training_tools,read_jsonl,compact

class Cursor:
 def __init__(self,contract):self.contract=contract;self.update=0
 def state_dict(self):return {'contract':self.contract,'update':self.update}
 def load_state_dict(self,s):
  if s['contract']!=self.contract:raise ValueError('Recovery contract changed')
  self.update=s['update']

def main():
 p=argparse.ArgumentParser()
 for k in ('parent','data','prepared','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--mode',choices=['train','reload','preflight'],default='train')
 p.add_argument('--steps',type=int,default=200);p.add_argument('--world-size',type=int,default=4)
 p.add_argument('--resume',type=Path);a=p.parse_args()
 if a.steps!=200:raise ValueError('This pilot release authorizes exactly 200 updates')
 ready=json.loads((a.data/'READY.json').read_text())
 for n,h in ready.items():
  if sha256(a.data/n)!=h:raise ValueError('Pilot data changed')
 records=list(read_jsonl(a.data/'records.jsonl'))
 audit=json.loads((a.data/'AUDIT.json').read_text())
 if sha256(a.prepared/'READY.json')!=audit['source_ready_sha256']:raise ValueError('TRAIN registry/source snapshot changed')
 tools=load_training_tools(a.prepared/'train_tools.jsonl');hard=json.loads((a.prepared/'hard-negatives.json').read_text())
 if len(records)!=1600 or any(r['split']!='train' for r in records):raise ValueError('1600 TRAIN exposures required')
 parent=json.loads((a.parent/'agent.json').read_text())
 independent=json.loads((a.parent.parent.parent/'independent-reload.json').read_text())
 if not independent['passed'] or independent['checkpoint_manifest_sha256']!=sha256(a.parent/'SHA256.json'):raise ValueError('Parent reload proof mismatch')
 if parent['interface']!=PARENT_INTERFACE or parent['metadata']['updates']!=5200:raise ValueError('Expected v41-5200 parent')
 if a.mode=='preflight':
  from transformers import AutoTokenizer
  from latent_register.toolbench_curriculum import render_parts
  from latent_register.toolbench_topk import reader_parts
  tok=AutoTokenizer.from_pretrained(a.parent/'backbone',local_files_only=True)
  limits=parent['limits'];maximum=0;state_max=0
  for r in records:
   state=r.get('task_state_target');h=r['history'];visible=observed_history(h,tools) if state else h
   if state:validate_state(state,h)
   cond=visible+([{'type':'task_state_estimate','content':state,'warning':'Model estimate; verify against original request and observations. Not new evidence.'}] if state else [])
   selected=tools[r['selected']]
   tasks=[]
   if r['masks']['control']:tasks.append(('control','<tool_request>'))
   if r.get('intent_supervision'):tasks.append(('intent',r.get('next_intent','')))
   if r['masks']['answer']:tasks.append(('answer',r.get('answer','')))
   if r['masks']['arguments']:tasks.append(('arguments',compact(r['arguments'])))
   if state:tasks.append(('task_state',compact(state)))
   for task,target in tasks:
    context=visible if task=='task_state' else cond
    parts=render_parts(tok,context,task,condition='full_document',document=selected.registration_document,history_format=parent['history_format'],next_intent=r.get('next_intent','') if task=='arguments' else '')
    n=sum(map(len,parts))+len(tok.encode(target,add_special_tokens=False))+1
    if task=='task_state':state_max=max(state_max,len(tok.encode(target,add_special_tokens=False))+1)
    if len(tok.encode(target,add_special_tokens=False))+1>limits['target']:raise ValueError('Target too long')
    maximum=max(maximum,n)
   parts=reader_parts(tok,cond,r.get('next_intent',''),5,parent['history_format']);maximum=max(maximum,sum(map(len,parts))+80+1)
   if r.get('repair'):
    rep=r['repair'];parts=render_parts(tok,cond,'repair',condition='full_document',document=selected.registration_document,previous=rep['previous'],error=rep['error'],history_format=parent['history_format'])
    maximum=max(maximum,sum(map(len,parts))+len(tok.encode(compact(rep['target']),add_special_tokens=False))+1)
  if maximum>limits['context'] or state_max>768:raise ValueError('Complete prefix/state exceeds budget')
  report={'passed':True,'records':len(records),'maximum_positions':maximum,'state_target_max_tokens':state_max,'data_sha256':sha256(a.data/'READY.json'),'parent_manifest_sha256':sha256(a.parent/'SHA256.json')}
  a.output.mkdir(parents=True,exist_ok=False);(a.output/'preflight.json').write_text(json.dumps(report,indent=2));print(report);return
 if a.mode=='reload':
  out=a.output
  completed=json.loads((out/'TRAINING_COMPLETE.json').read_text())
  if completed['pilot_updates']!=200:raise ValueError('Pilot incomplete')
  probe=json.loads((out/'train-probe.json').read_text())
  agent=load_agent(out/'checkpoint',device='cuda:0',torch_dtype=torch.bfloat16);agent.eval()
  with torch.no_grad():losses=agent([records[probe['index']]],tools,hard=hard,stage=2,candidate_count=16,seed=17)
  diffs={k:abs(float(v)-probe['losses'][k]) for k,v in losses.items()}
  ok=set(losses)==set(probe['losses']) and all(math.isfinite(v) and v<=.02 for v in diffs.values())
  report={'passed':ok,'absolute_loss_differences':diffs,'tolerance':.02,'checkpoint_manifest_sha256':sha256(out/'checkpoint/SHA256.json'),'pilot_updates':200,'optimizer_updates_during_reload':0}
  with (out/'independent-reload.json').open('x') as f:json.dump(report,f,indent=2)
  if not ok:raise RuntimeError('Independent reload gate failed')
  print(report);return
 acc=Accelerator(gradient_accumulation_steps=8//a.world_size,mixed_precision='no',kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,broadcast_buffers=False)])
 if acc.num_processes!=a.world_size or a.world_size not in (2,4):raise ValueError('Actual world size mismatch')
 set_seed(17)
 if acc.is_main_process:a.output.mkdir(parents=True,exist_ok=False)
 acc.wait_for_everyone()
 agent=load_agent(a.parent,device=str(acc.device),torch_dtype=torch.bfloat16);agent.__class__=TaskStateAgent
 agent.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 train_lora_and_compilers(agent)
 opt=torch.optim.AdamW(optimizer_groups(agent,2e-5,5e-5),weight_decay=.01)
 # New bounded fine-tuning recipe: 2000-step horizon, pause after 200; not an old-state resume.
 scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/50,1.)*.5*(1+math.cos(math.pi*min(s,2000)/2000)))
 model,opt=acc.prepare(agent,opt)
 contract={'interface':INTERFACE,'parent':sha256(a.parent/'SHA256.json'),'data':sha256(a.data/'READY.json'),'world_size':a.world_size,'global_batch':8,'horizon':2000,'pilot_steps':200,'lora_lr':2e-5,'compiler_lr':5e-5,'seed':17,'new_optimizer':True}
 cursor=Cursor(contract);acc.register_for_checkpointing(scheduler,cursor)
 if a.resume:
  receipt=json.loads((a.resume/'READY.json').read_text())
  for n,h in receipt['files'].items():
   if sha256(a.resume/n)!=h:raise ValueError('Recovery bytes changed')
  acc.load_state(str(a.resume))
 if acc.is_main_process:(a.output/'CONTRACT.json').write_text(json.dumps(contract,indent=2))
 started=time.monotonic();accum=8//a.world_size;opt.zero_grad()
 for update in range(cursor.update,200):
  for micro in range(accum):
   index=update*8+micro*a.world_size+acc.process_index
   with acc.accumulate(model):
    losses=model([records[index]],tools,hard=hard,stage=2,candidate_count=16,seed=17)
    bad=not torch.isfinite(losses['loss'])
    if acc.reduce(torch.tensor(int(bad),device=acc.device),reduction='sum').item():raise RuntimeError('Nonfinite training loss')
    acc.backward(losses['loss'])
    if acc.sync_gradients:acc.clip_grad_norm_(model.parameters(),1.)
    opt.step();opt.zero_grad()
  scheduler.step();cursor.update=update+1
  if acc.is_main_process:
   item={'pilot_updates':cursor.update,'parent_updates':5200,'training_exposures':cursor.update*8,'elapsed_seconds':time.monotonic()-started,'losses':{k:float(v.detach()) for k,v in losses.items()},'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
   with (a.output/'progress.jsonl').open('a') as f:f.write(compact(item)+'\n')
   tmp=a.output/'progress.tmp';tmp.write_text(compact(item));os.replace(tmp,a.output/'progress.json');print(compact(item),flush=True)
  budget_stop=bool(acc.reduce(torch.tensor(int(time.monotonic()-started>=39600),device=acc.device),reduction='sum').item())
  if cursor.update%100==0 or budget_stop:
   dest=a.output/'recovery'/f'update-{cursor.update}'
   acc.save_state(str(dest));acc.wait_for_everyone()
   if acc.is_main_process:
    hashes={str(p.relative_to(dest)):sha256(p) for p in dest.rglob('*') if p.is_file()}
    (dest/'READY.json').write_text(json.dumps({'updates':cursor.update,'files':hashes},indent=2))
   acc.wait_for_everyone()
  if budget_stop:break
 raw=acc.unwrap_model(model);raw.eval();index=next(i for i,r in enumerate(records) if r.get('task_state_target') and r['masks']['selection'])
 with torch.no_grad():probe=raw([records[index]],tools,hard=hard,stage=2,candidate_count=16,seed=17)
 if acc.is_main_process:
  (a.output/'train-probe.json').write_text(json.dumps({'index':index,'losses':{k:float(v) for k,v in probe.items()}},indent=2))
  save_agent(raw,a.output/'checkpoint',metadata={'official_base_fresh_start':False,'warm_start_parent_manifest':contract['parent'],'updates':cursor.update,'parent_updates':5200,'train_api_identities':sorted(tools),'contract':contract,'stop_reason':'pilot_complete' if cursor.update==200 else 'time_budget_pause'})
  (a.output/('TRAINING_COMPLETE.json' if cursor.update==200 else 'TRAINING_INCOMPLETE.json')).write_text(json.dumps({'pilot_updates':cursor.update,'parent_updates':5200,'status':'exported_reload_pending'},indent=2))
 acc.wait_for_everyone()
if __name__=='__main__':main()
