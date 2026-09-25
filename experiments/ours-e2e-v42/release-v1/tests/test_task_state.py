import copy,json,unittest
from unittest.mock import patch
import torch
from test_topk import fixture
from latent_register.toolbench_task_state import TaskStateAgent,validate_state,teacher_payload,observed_history
from latent_register.toolbench_hybrid import HybridDocumentAgent
from latent_register.toolbench_agent import Generation
from latent_register.toolbench_data import ToolSpec,FINISH

class StateTests(unittest.TestCase):
 def fixture(self):
  a,t,r=fixture();a.__class__=TaskStateAgent;a.eval()
  s={'requirements':[{'quote':'search Paris','status':'pending','evidence':[],'note':'Need search results'}]}
  return a,t,r,s
 def test_teacher_excludes_targets_and_future(self):
  a,t,r,s=self.fixture();p=teacher_payload(r,t)
  r.update(selected='FUTURE',answer='SECRET',next_intent='HIDDEN',repair={'target':'SECRET'})
  self.assertEqual(p,teacher_payload(r,t));self.assertNotIn('api00',json.dumps(p))
 def test_future_observation_and_invented_requirement_rejected(self):
  a,t,r,s=self.fixture();s['requirements'][0].update(status='supported',evidence=[{'observation':1,'quote':'Paris'}])
  with self.assertRaises(ValueError):validate_state(s,r['history'])
  s['requirements'][0].update(quote='invented',status='pending',evidence=[])
  with self.assertRaises(ValueError):validate_state(s,r['history'])
 def test_quoted_observation_and_error_distinction(self):
  a,t,r,s=self.fixture();h=r['history']+[{'type':'observation','content':{'error':'','response':'Paris found'}}]
  s['requirements'][0].update(status='supported',evidence=[{'observation':1,'quote':'Paris found'}])
  self.assertEqual(validate_state(s,h),s)
  h[-1]['content']['error']='execution failed'
  with self.assertRaises(ValueError):validate_state(s,h)
 def test_repair_event_preserved_and_tools_bound(self):
  a,t,r,s=self.fixture();h=r['history']+[{'type':'call','api_identity':'api00','arguments':{}},{'type':'validation_error','error':'required text'}]
  out=observed_history(h,t)
  self.assertEqual(out[-1],h[-1]);self.assertEqual(out[-2]['tool_description_excerpt'],t['api00'].document)
  self.assertNotIn('api01',json.dumps(out))
 def test_forward_adds_real_state_gradient_and_restores_context(self):
  a,t,r,s=self.fixture();r['task_state_target']=s
  result=a([r],t,stage=2)
  self.assertIn('task_state',result);self.assertGreater(float(result['task_state']),0)
  result['task_state'].backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in a.backbone.parameters()))
  self.assertIsNone(a._task_state)
 def test_replay_is_numerically_same_as_hybrid(self):
  a,t,r,s=self.fixture();b=copy.deepcopy(a);b.__class__=HybridDocumentAgent
  x=a([r],t,stage=2);y=b([r],t,stage=2)
  for k in y:torch.testing.assert_close(x[k],y[k])
  self.assertEqual(float(x['task_state']),0)
 def test_state_rendered_in_reader_but_not_document_readback(self):
  a,t,r,s=self.fixture();a._task_state=s
  from latent_register.toolbench_topk import reader_parts
  with patch('latent_register.toolbench_topk.reader_parts',wraps=reader_parts) as call:
   a.reader_prefix(r['history'],'search',torch.randn(5,16,32))
  self.assertEqual(call.call_args.args[1][-1]['type'],'task_state_estimate')
  from latent_register.toolbench_curriculum import render_parts
  with patch('latent_register.toolbench_curriculum.render_parts',wraps=render_parts) as call:
   a.prefix([], 'information',memory=torch.randn(16,32))
  self.assertEqual(call.call_args.args[1],[])
 def test_checkpoint_roundtrip_real_qwen(self):
  import tempfile
  from pathlib import Path
  from latent_register.toolbench_checkpoint import save_agent,load_agent
  a,t,r,s=self.fixture();r['task_state_target']=s
  with tempfile.TemporaryDirectory() as root:
   path=Path(root)/'checkpoint';save_agent(a,path,metadata={'test':True})
   b=load_agent(path,device='cpu',torch_dtype=torch.float32)
   self.assertIsInstance(b,TaskStateAgent)
   with torch.no_grad():x=a([r],t,stage=2);y=b([r],t,stage=2)
   for k in x:torch.testing.assert_close(x[k],y[k])
 def test_runtime_and_training_condition_match(self):
  a,t,r,s=self.fixture();r['task_state_target']=s
  captured=[]
  original=a.prefix
  def capture(history,task,*args,**kwargs):
   result=original(history,task,*args,**kwargs)
   if task=='control':captured.append(result.detach().clone())
   return result
  with patch.object(a,'prefix',side_effect=capture):a([r],t,stage=2)
  t[FINISH]=ToolSpec(FINISH,'finish',{'type':'object','properties':{}})
  with patch.object(a,'prefix',side_effect=capture),patch.object(a,'register_tools'),patch.object(a,'generate',side_effect=[Generation(json.dumps(s),(),True),Generation('<give_up>',(),True)]):a.decide(r['history'],t)
  torch.testing.assert_close(captured[0],captured[1])
 def test_runtime_state_before_control_and_invalid_state_fallback(self):
  for text in ['bad json',json.dumps(self.fixture()[3])]:
   a,t,r,s=self.fixture();t[FINISH]=ToolSpec(FINISH,'finish',{'type':'object','properties':{}})
   generated=[Generation(text,(),True),Generation('<final>',(),True),Generation('No information yet',(),True)]
   with patch.object(a,'register_tools'),patch.object(a,'generate',side_effect=generated):d=a.decide(r['history'],t)
   self.assertEqual(d.generation_budgets['task_state']['valid_provenance'],text!='bad json')
   self.assertIsNone(a._task_state)
if __name__=='__main__':unittest.main()
