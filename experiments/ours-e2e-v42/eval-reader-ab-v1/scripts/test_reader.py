import unittest,types,torch
from reader_variant import install,project
class Tokenizer:
    last=None
    def encode(self,text,add_special_tokens=False):
        if text in ['1','2','3','4','5']:return [int(text)]
        return [6]*len(text)
    def apply_chat_template(self,messages,**kwargs):self.last=messages;return messages[-1]['content']
class Backbone:
    def __init__(self):
        self.e=torch.nn.Embedding(8,2);self.h=torch.nn.Linear(2,8,bias=False)
        with torch.no_grad():self.h.weight.zero_();self.h.weight[3,0]=1
    def get_output_embeddings(self):return self.h
    def get_input_embeddings(self):return self.e
class Agent:
    def __init__(self):
        self.training=False;self.history_format='legacy';self.limits=types.SimpleNamespace(context=10000);self.tokenizer=Tokenizer();self.backbone=Backbone()
    def _ids(self,x):return torch.tensor(x,dtype=torch.long)
    def _hidden(self,**kwargs):return types.SimpleNamespace(last_hidden_state=torch.tensor([[[1.,0.]]]))
    def reader_prefix(self,*args):return torch.zeros(1,3,2)
    def _with_state(self,h):return h+[{'type':'task_state_estimate','content':'pending'}]
class Tests(unittest.TestCase):
    def setUp(self):self.tools={str(i):types.SimpleNamespace(registration_document=f'DOC{i} COMPLETE SCHEMA',document_hash=str(i)*64) for i in range(5)}
    def test_autocast_and_grad(self):
        h=torch.tensor([[1.,.125]],requires_grad=True);w=torch.tensor([[13.,.01],[13.,.02]],requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):s=project(h,w)
        self.assertEqual(s.dtype,torch.float32);self.assertGreater(s[0,1],s[0,0]);s.sum().backward();self.assertTrue(torch.isfinite(w.grad).all())
    def test_full_docs_exact_set(self):
        a=install(Agent(),self.tools,'documents_fp32')
        with torch.no_grad():selected,b=a.select_from_ranking([{'type':'user','content':'REQUEST'}],'INTENT',tuple(self.tools))
        self.assertEqual(selected,b['reader_candidates'][2]);text=a.tokenizer.last[-1]['content']
        for t in self.tools.values():self.assertIn(t.registration_document,text)
        self.assertIn('REQUEST',text);self.assertIn('INTENT',text);self.assertIn('pending',text);self.assertEqual(b['reader_memory_positions'],0)
    def test_no_truncation(self):
        a=install(Agent(),self.tools,'documents_fp32');a.limits.context=1
        with self.assertRaisesRegex(ValueError,'no truncation'):a.select_from_ranking([{'type':'user','content':'X'}],'x',tuple(self.tools))
    def test_no_document_training(self):
        a=install(Agent(),self.tools,'documents_fp32');a.training=True
        with self.assertRaises(ValueError):a.select_from_ranking([],'x',tuple(self.tools))
    def test_no_double_install(self):
        a=install(Agent(),self.tools,'memory_fp32')
        with self.assertRaises(ValueError):install(a,self.tools,'documents_fp32')
    def test_memory_projection(self):
        a=install(Agent(),self.tools,'memory_fp32')
        with torch.no_grad():scores=a.reader_scores([],'x',torch.zeros(5,16,2))
        self.assertEqual(scores.argmax().item(),2);self.assertEqual(len(a._last_reader_logits),5)
if __name__=='__main__':unittest.main()
