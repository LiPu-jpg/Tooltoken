import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_topk import fixture
from latent_register.toolbench_hybrid import HybridDocumentAgent,INTERFACE,RECIPE
from latent_register.toolbench_topk import INTERFACE as OLD,RECIPE as TOPK
from latent_register.toolbench_curriculum import CurriculumAgent
from latent_register.toolbench_agent import Generation,GenerationFailure
from latent_register.toolbench_data import FINISH,ToolSpec
from latent_register.hybrid_transition import HybridTrainingState
from latent_register.selection_candidates import fingerprint
from latent_register.toolbench_checkpoint import save_agent,load_agent

class HybridTests(unittest.TestCase):
 def fixture(self):
  a,t,r=fixture();a.__class__=HybridDocumentAgent;a.eval();return a,t,r
 def test_args_and_repair_full_document_info_stays_memory(self):
  a,t,r=self.fixture();m=torch.randn(16,32)
  for task in ('arguments','repair','information'):
   with patch.object(CurriculumAgent,'prefix',wraps=CurriculumAgent.prefix.__get__(a)) as call:
    x=a.prefix(r['history'] if task!='information' else [],task,memory=m,document=t['api00'].registration_document)
   self.assertEqual(call.call_args.kwargs['condition'],'full_document' if task!='information' else None)
   self.assertGreater(x.shape[1],0)
  with self.assertRaises(ValueError):a.prefix(r['history'],'arguments',memory=m)
 def test_real_forward_args_independent_of_memory_but_reader_trains_memory(self):
  a,t,r=self.fixture();out=a([r],t,stage=2)
  self.assertEqual(set(out),{'selection','arguments','information','control','repair','answer','intent','reader','loss'})
  out['arguments'].backward(retain_graph=True)
  self.assertFalse(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.memory_compiler.parameters()))
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.backbone.parameters()))
  a.zero_grad();out['reader'].backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.memory_compiler.parameters()))
 def test_failed_arguments_keep_reader_telemetry(self):
  a,t,r=self.fixture();t[FINISH]=ToolSpec(FINISH,'finish',{'type':'object','properties':{}})
  ids=sorted(set(t)-{FINISH});a._registry={k:(None,torch.ones(32)*i,torch.ones(16,32)*i) for i,k in enumerate(ids)}
  outputs=[Generation('<tool_request>',(),True),Generation('search Paris',(),True),Generation('{}',(),True)]
  with patch.object(a,'register_tools'),patch.object(a,'selection_scores',return_value=torch.arange(len(ids)).reshape(1,-1).float()),patch.object(a,'generate',side_effect=outputs):
   with self.assertRaises(GenerationFailure) as ctx:a.decide(r['history'],t)
  self.assertEqual(len(ctx.exception.details['reader_trace']['logits']),5)
  self.assertEqual(ctx.exception.details['reader_trace']['reader_memory_positions'],80)
  self.assertEqual(len(ctx.exception.details['selected_document_sha256']),64)
 def test_roundtrip_same_state_no_new_parameters(self):
  a,t,r=self.fixture();m=torch.randn(5,16,32);old,_,_=fixture()
  self.assertEqual(set(a.state_dict()),set(old.state_dict()))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'export';save_agent(a,p,metadata={'hybrid_document_reader':RECIPE,'topk_reader':TOPK})
   b=load_agent(p);self.assertIsInstance(b,HybridDocumentAgent)
   torch.testing.assert_close(a.reader_scores(r['history'],'',m),b.reader_scores(r['history'],'',m))
   torch.testing.assert_close(a.prefix(r['history'],'arguments',document=t['api00'].registration_document),b.prefix(r['history'],'arguments',document=t['api00'].registration_document))
 def test_transition_strict_and_future_resume(self):
  a,t,r=self.fixture();old={'interface':OLD,'topk_reader':TOPK,'world_size':4,'schedule':'fixed'};new={**old,'interface':INTERFACE,'hybrid_document_reader':RECIPE}
  cursor={'updates':5000,'next_local_microbatch':12000,'stage':2}
  saved={'contract':old,'cursor':cursor,'backbone_buffers':{k:v.clone() for k,v in a.backbone.named_buffers()}}
  transition={'kind':'selected-document-v1','parent_cursor':cursor,'at_update':5000,'parent_contract_sha256':fingerprint(old)}
  state=HybridTrainingState(a,new,transition);state.load_state_dict(saved);self.assertTrue(state.transition_applied);self.assertEqual(state.cursor,cursor)
  for bad in ({**new,'world_size':2},{**new,'schedule':'changed'}):
   with self.assertRaises(ValueError):HybridTrainingState(a,bad,transition).load_state_dict(saved)
  with self.assertRaises(ValueError):HybridTrainingState(a,new).load_state_dict(saved)
  HybridTrainingState(a,new).load_state_dict({**saved,'contract':new})
if __name__=='__main__':unittest.main()
