"""Retain official labels; apply reviewed causal state with explicit conflict masks."""
import argparse,copy,json,hashlib,collections
from pathlib import Path
from latent_register.toolbench_task_state import validate_state
from latent_register.toolbench_data import compact
p=argparse.ArgumentParser()
for key in ('source','annotations','output'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--tokenizer',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
x=json.loads(a.source.read_text());records=[];decisions=[];counts=collections.Counter()
for i,original in enumerate(x['records']):
 r=copy.deepcopy(original);receipt=a.annotations/f'{i:05}.json';item={'index':i,'record_id':r['id']}
 if receipt.exists():
  label=json.loads(receipt.read_text())
  if label.get('valid') and i not in {67,79}: # manually audited uncertainty/coverage conflicts; preserve raw receipt
   state=validate_state(label['state'],r['history'])
   if len(tok.encode(compact(state),add_special_tokens=False))+1<=768:
    # Review estimate is not ground truth: avoid contradictory supervision, retain record and reason.
    statuses=[q['status'] for q in state['requirements']]
    if r['mode']=='final' and any(s!='supported' for s in statuses):
     r['masks']['control']=0;r['masks']['answer']=0;item['conflict_mask']='final_without_all_requirements_supported'
    elif r['mode']=='give_up' and all(s=='supported' for s in statuses):
     r['masks']['control']=0;item['conflict_mask']='give_up_with_supported_requirements'
    if i%4!=0:
     r['task_state_target']=state;item['state_supervision']=True
    else:item['state_supervision']=False;item['reason']='fixed_25pct_original_interface_replay'
   else:item['reason']='state_exceeds_generation_budget'
  else:item['reason']='review_invalid'
 else:item['reason']='no_reviewed_valid_annotation'
 counts['state_supervision' if r.get('task_state_target') else 'original_interface_replay']+=1
 if item.get('conflict_mask'):counts[item['conflict_mask']]+=1
 records.append(r);decisions.append(item)
if counts['state_supervision']<500:raise ValueError('Too few reviewed causal labels for pilot; no training')
for name,rows in [('records.jsonl',records),('decisions.jsonl',decisions)]:
 with (a.output/name).open('x') as f:
  for r in rows:f.write(compact(r)+'\n')
audit={'source_ready_sha256':x['source_ready_sha256'],'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'counts':dict(counts),'records':len(records),'teacher_review_is_not_truth':True,'teacher_revision':'v2','split':'train','selection_seed':x['selection_seed'],'new_optimizer':True,'purpose':'200-update causal state pilot, not full-dataset retraining'}
(a.output/'AUDIT.json').write_text(json.dumps(audit,indent=2))
ready={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()}
(a.output/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps(audit))
