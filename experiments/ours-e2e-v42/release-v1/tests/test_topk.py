import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
from latent_register.toolbench_topk import TopKCurriculumAgent,shortlist,candidate_order,INTERFACE,RECIPE
from latent_register.toolbench_curriculum import INTERFACE as OLD_INTERFACE
from latent_register.toolbench_lora import OptimizedCurriculumAgent
from latent_register.toolbench_data import ToolSpec,FINISH
from latent_register.toolbench_agent import Limits,Generation
from latent_register.topk_transition import TopKTrainingState
from latent_register.selection_candidates import fingerprint


def fixture():
 torch.set_num_threads(1);torch.manual_seed(17)
 words=['[UNK]','[EOS]','[PAD]','Paris','search','1','2','3','4','5']+[f'api{i:02}' for i in range(20)]
 vocab={w:i for i,w in enumerate(words)}
 engine=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));engine.pre_tokenizer=Whitespace()
 tok=PreTrainedTokenizerFast(tokenizer_object=engine,unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
 tok.chat_template="{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
 base=Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab),hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096))
 agent=TopKCurriculumAgent(base,tok,rank=8,slots=16,memory_width=32,memory_heads=4,memory_depth=4,limits=Limits(4096,1024,512))
 schema={'type':'object','properties':{'text':{'type':'string'}},'required':['text']}
 tools={f'api{i:02}':ToolSpec(f'api{i:02}',f'search api{i:02}',schema) for i in range(20)}
 record={'id':['train',0],'selected':'api00','history':[{'type':'user','content':'search Paris'}], 'arguments':{'text':'Paris'},'mode':'tool','masks':{'selection':1,'arguments':1,'control':1,'answer':0},'repair':{'previous':'{}','error':'text is required','target':{'text':'Paris'}},'next_intent':'search Paris','intent_supervision':True}
 return agent,tools,record

class TopKTests(unittest.TestCase):
 def test_training_gold_insertion_runtime_no_insertion(self):
  ids=[f'api{i:02}' for i in range(16)];scores=torch.arange(16.).reshape(1,-1)
  runtime,present=shortlist(scores,ids,[],gold=None)
  self.assertEqual(set(runtime),set(ids[-5:]));self.assertIsNone(present)
  training,present=shortlist(scores,ids,[],gold='api00')
  self.assertFalse(present);self.assertIn('api00',training);self.assertEqual(len(set(training)),5)
  self.assertEqual(candidate_order([],training),training)

 def test_reader_all_80_positions_receive_gradient(self):
  a,tools,r=fixture();a.eval()
  memories=torch.randn(5,16,32,requires_grad=True)
  loss=torch.nn.functional.cross_entropy(a.reader_scores(r['history'],'search Paris',memories),torch.tensor([2]))
  loss.backward()
  self.assertTrue(torch.isfinite(memories.grad).all())
  self.assertTrue((memories.grad.abs().sum(-1)>0).all())

 def test_real_forward_preserves_losses_and_trains_memory(self):
  a,tools,r=fixture();a.eval()
  with patch.object(a,'compile',wraps=a.compile) as compile_call:
   result=a([r],tools,stage=2)
  self.assertEqual(len(compile_call.call_args_list[0].kwargs['memory_indices']),16)
  baseline=copy.deepcopy(a);baseline.__class__=OptimizedCurriculumAgent
  old=baseline([r],tools,stage=2)
  for key in old:
   if key!='loss':torch.testing.assert_close(result[key],old[key],rtol=1e-4,atol=1e-4)
  torch.testing.assert_close(result['loss'],old['loss']+result['reader'],rtol=1e-4,atol=1e-4)
  result['reader'].backward()
  self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in a.memory_compiler.parameters()))
  self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in a.backbone.parameters()))
  self.assertEqual(set(a.state_dict()),set(baseline.state_dict()))

 def test_runtime_can_select_non_top1_and_uses_its_memory(self):
  a,tools,r=fixture();a.eval();tools[FINISH]=ToolSpec(FINISH,'finish',{'type':'object','properties':{}})
  ids=sorted(k for k in tools if k!=FINISH)
  a._registry={k:(None,torch.ones(32)*i,torch.ones(16,32)*i) for i,k in enumerate(ids)}
  ranked=list(reversed(ids));chosen=candidate_order(r['history'],ranked[:5]);target=next(i for i,k in enumerate(chosen) if k!=ranked[0])
  logits=torch.full((1,5),-10.);logits[0,target]=10.
  outputs=[Generation('<tool_request>',(),True),Generation('search Paris',(),True),Generation('{"text":"Paris"}',(),True)]
  with patch.object(a,'register_tools'),patch.object(a,'selection_scores',return_value=torch.arange(len(ids)).reshape(1,-1).float()),patch.object(a,'reader_scores',return_value=logits) as reader,patch.object(a,'generate',side_effect=outputs),patch.object(a,'prefix',wraps=a.prefix) as prefix:
   d=a.decide(r['history'],tools)
  self.assertEqual(d.api_identity,chosen[target]);self.assertNotEqual(d.api_identity,ranked[0])
  self.assertEqual(reader.call_args.args[2].shape,(5,16,32))
  torch.testing.assert_close(prefix.call_args.kwargs['memory'],a._registry[d.api_identity][2])
  self.assertEqual(d.generation_budgets['reader_selected_slot'],target+1)

 def test_no_future_target_in_reader(self):
  a,_,r=fixture();mem=torch.randn(5,16,32)
  p=a.reader_prefix(r['history'],r['next_intent'],mem)
  r['arguments']={'secret_future':'hidden'};r['answer']='hidden'
  torch.testing.assert_close(p,a.reader_prefix(r['history'],r['next_intent'],mem))

 def test_checkpoint_roundtrip_preserves_reader(self):
  from latent_register.toolbench_checkpoint import save_agent,load_agent
  a,_,r=fixture();a.eval();mem=torch.randn(5,16,32)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'checkpoint';save_agent(a,path,metadata={'topk_reader':RECIPE})
   b=load_agent(path)
   self.assertIsInstance(b,TopKCurriculumAgent)
   torch.testing.assert_close(a.reader_scores(r['history'],'',mem),b.reader_scores(r['history'],'',mem))

 def test_transition_rejects_other_changes_restores_cursor(self):
  a,_,_=fixture();old={'interface':OLD_INTERFACE,'schedule':'fixed','world_size':4}
  cursor={'updates':4200,'next_local_microbatch':10400,'stage':2}
  saved={'contract':old,'cursor':cursor,'backbone_buffers':{k:v.clone() for k,v in a.backbone.named_buffers()}}
  transition={'kind':'topk-reader-v1','at_update':4200,'parent_contract_sha256':fingerprint(old),'parent_cursor':cursor}
  new={**old,'interface':INTERFACE,'topk_reader':RECIPE}
  s=TopKTrainingState(a,new,transition);s.load_state_dict(saved)
  self.assertTrue(s.transition_applied);self.assertEqual(s.cursor,cursor)
  for bad in ({**new,'world_size':2},{**new,'schedule':'new'}):
   with self.assertRaises(ValueError):TopKTrainingState(a,bad,transition).load_state_dict(saved)
  with self.assertRaises(ValueError):TopKTrainingState(a,new).load_state_dict(saved)

if __name__=='__main__':unittest.main()
