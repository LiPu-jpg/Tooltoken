"""Actual frozen renderer + adapter import, no model weights or GPU calls."""
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
V=ROOT.parents[1]
sys.path.insert(0,str(V/'release-v1/source/src'))
from latent_register.toolbench_curriculum import render_parts
from latent_register.toolbench_task_state import observed_history
from latent_register.toolbench_data import compact

spec=importlib.util.spec_from_file_location('v6_entry',ROOT/'scripts/run_native_variant.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        self.body=messages[-1]['content'];return self.body
    def encode(self,text,**kwargs):return list(text.encode('utf-8'))

class RuntimeTests(unittest.TestCase):
    def test_real_renderers_retain_query_constraints_and_unexecuted_feedback(self):
        query='We are six people. Need services on 2030-05-01. Include 北京 departure stations.'
        history=[{'type':'user','content':query},
                 {'type':'call','api_identity':'t','arguments':{'people':6}},
                 {'type':'observation','api_identity':'t','content':{'response':{'departure':'09:10'},'error':''}},
                 {'type':'recovery_feedback','is_observation':False,
                  'content':{'plan':'Use a service endpoint, not just the station listing.',
                             'rejected_proposals':[{'tool':'u','arguments':{},'executed':False}]}}]
        before=copy.deepcopy(history)
        visible=observed_history(history,{'t':SimpleNamespace(document='TRAIN service endpoint')})
        for task in ('control','intent','answer','task_state','arguments'):
            tok=Tokenizer()
            parts=render_parts(tok,visible,task,condition='full_document',document='TRAIN service endpoint',
                               next_intent='Find matching six-person services.' if task=='arguments' else '')
            self.assertEqual(len(parts),1)
            self.assertIn(compact(query),tok.body)
            self.assertIn('"type":"recovery_feedback","is_observation":false',tok.body)
            self.assertEqual(tok.body.count('"type":"observation"'),1)
            self.assertIn('"observation_id":1',tok.body)
            if task=='arguments':self.assertIn('TRAIN service endpoint',tok.body)
        self.assertEqual(history,before)

    def test_missing_executor_receipt_and_parser_rejection_are_not_erased(self):
        result={'status':'give_up_and_restart','history':[],'reason':'test','ledger':{},
                'budget':{'generations':2,'calls':1,'tokens':100},
                'events':[{'action':'rejected_generation_arguments'}],
                'requirement_extraction_fallback':True,'requirement_extraction_mode':'test',
                'native_diagnostics':[{'kind':'native_generation','seconds':1.5},
                                      {'kind':'auxiliary_generation','seconds':0.5}]}
        trace=entry.normalize_trace(result,[],4,{'seconds':1,'succeeded':0,'failed':0})
        self.assertEqual(trace['costs']['executions_without_receipt'],1)
        self.assertEqual(trace['costs']['validation_failures'],1)
        self.assertEqual(trace['costs']['generation_seconds'],2)
        self.assertEqual(trace['costs']['model_seconds_definition'],'legacy_episode_wall_including_executor')
        self.assertFalse(trace['task_success_judged'])

    def test_all_recorded_frozen_parent_sources_unchanged(self):
        manifest=json.loads((ROOT/'PARENT_SHA256.json').read_text())
        for relative,expected in manifest.items():
            self.assertEqual(hashlib.sha256((V/relative).read_bytes()).hexdigest(),expected,relative)

if __name__=='__main__':unittest.main()
