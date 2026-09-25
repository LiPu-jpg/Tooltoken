import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'variant/task_completion'))
from completion import Ledger, Budget, CompletionController, BoundedSession, deterministic_requirements
from evidence_patch import repair, catalogue, apply_plan
from runner import run
from native_adapter import NativeCallbacks, run_native, FINISH

APPROVE={'matches_request':True,'preserves_supported':True,'no_contradiction':True}
Q='List the departure stations and services for six people on 2030-05-01.'


def ledger(query=Q, response=None):
    obj=Ledger(query,[{'source_span':[0,len(query)],'constraints':{}}])
    obj.sync([{'type':'user','content':query},
              {'type':'call','api_identity':'train-test-service','arguments':{'people':6,'date':'2030-05-01'}},
              {'type':'observation','api_identity':'train-test-service',
               'content':{'error':'','response':response if response is not None else {'departure':'09:10','price':'40 EUR'}}}])
    return obj


def generation(plan, review=APPROVE):
    calls=[]
    def gen(purpose,payload,limit):
        calls.append((purpose,copy.deepcopy(payload),limit))
        return plan if purpose=='select_evidence_patch' else review
    return gen,calls


class EvidenceTests(unittest.TestCase):
    def test_acquired_service_added_without_losing_station_answer(self):
        l=ledger();cards,_=catalogue(l);text=cards['o2']['source_text']
        gen,calls=generation({'append':[{'observation':'o2','quote':text}],'replace':[]})
        before='Departure station: Harbor.'
        after,event=repair(before,l,gen)
        self.assertEqual(after,before+'\n\n'+text)
        self.assertEqual(event['outcome'],'source_patch_applied')
        self.assertFalse(event['semantic_success_certified'])
        self.assertEqual([x[0] for x in calls],['select_evidence_patch','review_evidence_patch'])

    def test_reassigned_entity_cannot_be_invented_by_rewriter(self):
        l=ledger(response='{"season":1960,"team":"Historical Team","points":48}')
        gen,calls=generation({'append':[{'observation':'o2','quote':'Modern Team: 48 points'}],'replace':[]})
        before='Existing supported answer.'
        after,event=repair(before,l,gen)
        self.assertEqual(after,before);self.assertEqual(event['outcome'],'invalid_patch_preserved')
        self.assertEqual(len(calls),1)

    def test_contradictory_paragraph_can_be_corrected_without_rewriting_other_paragraph(self):
        l=ledger(response='Logout result: successful.')
        before='Logout failed.\n\nOrder AB12 is delivered.'
        gen,_=generation({'append':[],'replace':[{'paragraph':'p1','observation':'o2','quote':'Logout result: successful.'}]})
        after,event=repair(before,l,gen)
        self.assertEqual(after,'Logout result: successful.\n\nOrder AB12 is delivered.')
        self.assertEqual(event['answer_before'],before)

    def test_destructive_or_unrelated_patch_rejected_by_review_retains_original(self):
        l=ledger(response='Other entity: 48 points.')
        for key in APPROVE:
            review=dict(APPROVE);review[key]=False
            gen,_=generation({'append':[],'replace':[{'paragraph':'p1','observation':'o2','quote':'Other entity: 48 points.'}]},review)
            after,event=repair('Correct entity: 40 points.',l,gen)
            self.assertEqual(after,'Correct entity: 40 points.')
            self.assertEqual(event['outcome'],'review_rejected_preserved')

    def test_missing_or_string_boolean_review_does_not_approve(self):
        l=ledger(response='Departure 09:10.')
        for review in (None, {}, 'invalid', dict(APPROVE,matches_request='true')):
            gen,_=generation({'append':[{'observation':'o2','quote':'Departure 09:10.'}],'replace':[]},review)
            self.assertEqual(repair('Harbor station.',l,gen)[0],'Harbor station.')

    def test_source_change_during_review_is_detected(self):
        l=ledger(response='Departure 09:10.')
        def gen(purpose,payload,limit):
            if purpose=='select_evidence_patch':
                return {'append':[{'observation':'o2','quote':'Departure 09:10.'}],'replace':[]}
            l.observations['o2']['sha256']='changed';return APPROVE
        after,event=repair('Station.',l,gen)
        self.assertEqual(after,'Station.');self.assertEqual(event['outcome'],'review_unavailable_preserved')

    def test_failed_envelope_retained_for_review_not_claimed_success(self):
        l=ledger(response='No authenticated session.')
        l.observations['o2']['content']['error']='Authorization required'
        cards,_=catalogue(l)
        self.assertEqual(cards['o2']['error_envelope'],'Authorization required')
        self.assertNotIn('task_success',cards['o2'])

    def test_empty_default_result_zero_false_remain_available(self):
        for value in ([],{},0,False):
            cards,_=catalogue(ledger(response=value))
            self.assertIn('o2',cards)
            self.assertEqual(json.loads(cards['o2']['source_text']),value)

    def test_binary_and_large_sources_omitted_explicitly_not_sliced(self):
        for value,reason in [('PNG\x00binary','binary_or_control_characters'),('x'*9000,'source_too_long')]:
            cards,omitted=catalogue(ledger(response=value))
            self.assertEqual(cards,{});self.assertEqual(omitted[0]['reason'],reason)

    def test_candidate_output_cap_does_not_truncate_original(self):
        l=ledger(response='Departure 09:10.');gen,calls=generation({'append':[{'observation':'o2','quote':'Departure 09:10.'}],'replace':[]})
        before='Long but valid original.'
        after,event=repair(before,l,gen,answer_fits=lambda _:False)
        self.assertEqual(after,before);self.assertEqual(event['outcome'],'answer_budget_preserved')
        self.assertEqual(len(calls),1)

    def test_omitted_review_budget_preserves_existing_answer(self):
        l=ledger(response='Departure 09:10.');gen,calls=generation({'append':[{'observation':'o2','quote':'Departure 09:10.'}],'replace':[]})
        c=CompletionController(l,Budget(8,1,1024));s=BoundedSession(c,gen)
        result=s.preserve_answer('Station.','test')
        self.assertEqual(result['answer'],'Station.')
        self.assertEqual(c.budget.tokens,1024);self.assertEqual(len(calls),1)
        self.assertEqual(s.patch_events[0]['outcome'],'review_unavailable_preserved')


class PipelineTests(unittest.TestCase):
    def test_parsed_coverage_patch_capacity_is_charged_once_and_review_not_free(self):
        l=ledger(response='Departure 09:10.')
        gen,calls=generation({'append':[{'observation':'o2','quote':'Departure 09:10.'}],'replace':[]})
        controller=CompletionController(l,Budget(8,2,1792))
        session=BoundedSession(controller,gen)
        audit={'requirements':[{'id':'r1','observation_status':'present','answer_status':'omitted'}]}
        result=session.finish_coverage('Station.',audit)
        self.assertEqual(result['answer'],'Station.\n\nDeparture 09:10.')
        self.assertEqual(controller.budget.tokens,1792)
        self.assertEqual(controller.budget.generations,2)
        self.assertEqual([c[2] for c in calls],[1024,768])
        self.assertEqual(controller.answer_repairs,1)

    def test_invalid_generated_arguments_record_rejection_without_execution(self):
        proposals=iter([{'kind':'invalid_tool','tool':'t','arguments_raw':'{broken',
                         'requirement_ids':['r1'],'reason':'invalid JSON'},
                        {'kind':'answer','answer':'Cannot retrieve this yet.'}])
        r=run(Q,[{'source_span':[0,len(Q)],'constraints':{}}],lambda c,g:next(proposals),lambda *a:None,
              lambda *a:self.fail('invalid input must not execute'),lambda *a:None,
              budget=Budget(8,55,20064),max_decisions=2)
        event=next(e for e in r['events'] if e['action']=='rejected_generation_arguments')
        self.assertEqual(event['arguments_raw'],'{broken')
        self.assertFalse(event['is_observation']);self.assertEqual(r['budget']['calls'],0)

    def test_fallback_retains_antecedent_quantity_time_unicode_and_does_not_certify(self):
        for query in ['I need mountain rainfall for the past 14 days. Can you provide this information?',
                      'List teams. More than 8100 are needed. Also show the first 30.',
                      'A value is 2.75. Show it, please. \u5317\u4eac\u3002']:
            definition=deterministic_requirements(query)
            l=Ledger(query,definition);l.coarse_fallback=True
            self.assertEqual(l.requirements['r1'].text,query)
            self.assertEqual(l.context()['original_query'],query)
            self.assertFalse(l.context()['requirements_are_atomic'])
            route=CompletionController(l,Budget(8,55,20064)).coverage_route('answer',{'requirements':[]})
            self.assertEqual(route['reason'],'unparsed_requirements_no_coverage_claim')

    def test_extract_failure_and_raw_reason_logged_original_history_unchanged(self):
        result=run(Q,None,lambda c,g:{'kind':'answer','answer':'Existing answer'},
                   lambda *a:'{"source_span":[0,3],"constraints":{}}',lambda *a:self.fail(),lambda *a:None,
                   budget=Budget(8,55,20064),max_decisions=1)
        self.assertEqual(result['history'],[{'type':'user','content':Q}])
        self.assertEqual(result['requirement_extraction_mode'],'full_query_native_planner_fallback')
        event=result['events'][0]
        self.assertEqual(event['action'],'requirement_extraction');self.assertFalse(event['accepted']);self.assertIn('raw',event)

    def test_giveup_uses_real_source_patch_not_unrelated_free_apology(self):
        proposals=iter([{'kind':'tool','tool':'t','arguments':{},'requirement_ids':['r1']},{'kind':'give_up'}])
        gen,_=generation({'append':[{'observation':'o2','quote':'Logout result: successful.'}],'replace':[]})
        r=run(Q,[{'source_span':[0,len(Q)],'constraints':{}}],lambda c,g:next(proposals),gen,
              lambda *a:{'content':{'error':'','response':'Logout result: successful.'},'transport':'ok'},lambda *a:None,
              budget=Budget(8,55,20064),max_decisions=3)
        self.assertEqual(r['answer'],'Logout result: successful.')
        self.assertEqual(r['budget']['calls'],1)
        self.assertFalse(r['semantic_success_certified'])

    def test_actual_duplicate_replan_is_logged_and_not_an_observation(self):
        def gen(purpose,payload,limit):
            return 'Choose another capability.' if purpose=='replan' else None
        r=run(Q,[{'source_span':[0,len(Q)],'constraints':{}}],
              lambda c,g:{'kind':'tool','tool':'t','arguments':{},'requirement_ids':['r1']},gen,
              lambda *a:{'content':{'error':'','response':'Found station.'},'transport':'ok'},lambda *a:None,
              budget=Budget(8,55,20064),max_decisions=4)
        self.assertEqual(r['budget']['calls'],1)
        self.assertTrue(any(e.get('action')=='replan' and e['text']=='Choose another capability.' for e in r['events']))
        self.assertEqual(sum(h['type']=='observation' for h in r['history']),1)


class FakeAgent:
    def __init__(self):
        self._task_state=None;self._inside_state_decision=False
        self.tokenizer=SimpleNamespace(encode=lambda s,**kw:list(s))
        self.seen=[]
    def generate(self,prefix,*,max_new_tokens):
        return SimpleNamespace(text='Native final.',truncated=False,token_ids=(1,2))
    def prefix(self,history,task):self.seen.append((copy.deepcopy(history),task));return None
    def decide(self,history,tools,**kwargs):
        self.seen.append((copy.deepcopy(history),self._inside_state_decision))
        self.generate(None,max_new_tokens=32)
        return SimpleNamespace(api_identity=FINISH,arguments={'return_type':'give_answer','final_answer':'Native final.'},
              thought='<final>',ranked_identities=('t','u'),generation_budgets={'next_intent':'Get required result.',
              'reader_candidates':['u','t'],'reader_logits':[0.1,0.8],'reader_selected_slot':2})


class AdapterTests(unittest.TestCase):
    def test_reader_logits_survive_later_argument_failure_and_wrapper_is_restored(self):
        a=FakeAgent()
        def select(history,intent,ranking):
            a._last_reader_logits=[0.1,0.8]
            return 't',{'reader_candidates':['u','t'],'reader_selected_slot':2}
        a.select_from_ranking=select
        def decide(history,tools,**kwargs):
            a.select_from_ranking(history,'Find the matching service.',['t','u'])
            exc=ValueError('Invalid JSON after selection')
            exc.details={'reason':'invalid_arguments','api_identity':'t','generated_text':'{broken'}
            raise exc
        a.decide=decide
        with NativeCallbacks(a,{FINISH:object(),'t':object()},Budget(8,55,20064),observe=lambda h,t:copy.deepcopy(h)) as cb:
            proposal=cb.propose(self.context(),None)
        row=next(d for d in cb.diagnostics if d['kind']=='reader_selection')
        self.assertEqual(row['intent'],'Find the matching service.')
        self.assertEqual(row['retrieval_top5'],['t','u'])
        self.assertEqual(row['metadata']['reader_candidates'],['u','t'])
        self.assertEqual(row['logits'],[0.1,0.8])
        self.assertEqual(row['selected_identity'],'t')
        self.assertEqual(proposal['kind'],'invalid_tool')
        self.assertIs(a.select_from_ranking,select)

    def context(self):
        l=ledger();l.coarse_fallback=True
        return {'history':[{'type':'user','content':Q}], 'completion':l.context(),'plan':None}

    def test_native_fallback_and_reader_diagnostics_preserved(self):
        a=FakeAgent();original=a.generate
        with NativeCallbacks(a,{FINISH:object(),'t':object()},Budget(8,55,20064),observe=lambda h,t:copy.deepcopy(h)) as cb:
            cb.propose(self.context(),None)
        self.assertFalse(a.seen[0][1]);self.assertEqual(a.generate,original)
        d=next(d for d in cb.diagnostics if d['kind']=='native_decision')
        self.assertEqual(d['generation_budgets']['reader_candidates'],['u','t'])
        self.assertEqual(d['ranked_identities_top5'],['t','u'])
        self.assertEqual(d['planner_mode'],'native')
        self.assertTrue(any(d['kind']=='native_generation' and d['actual_generated_tokens']==2 for d in cb.diagnostics))

    def test_force_finish_uses_answer_prefix_not_invalid_finish_only_decide(self):
        a=FakeAgent();a.decide=lambda *a,**kw:self.fail('must not use Finish-only registry')
        ctx=self.context();ctx['force_finish']=True
        with NativeCallbacks(a,{FINISH:object(),'t':object()},Budget(8,55,20064),observe=lambda h,t:copy.deepcopy(h)) as cb:
            proposal=cb.propose(ctx,None)
        self.assertEqual(proposal,{'kind':'answer','answer':'Native final.'})
        self.assertEqual(a.seen[0][1],'answer')

    def test_recovery_feedback_reaches_native_mode_without_forging_execution_history(self):
        a=FakeAgent();ctx=self.context();ctx['plan']='Try another operation.';before=copy.deepcopy(ctx['history'])
        with NativeCallbacks(a,{FINISH:object(),'t':object()},Budget(8,55,20064),observe=lambda h,t:copy.deepcopy(h)) as cb:
            cb.propose(ctx,None)
        self.assertEqual(ctx['history'],before)
        self.assertEqual(a.seen[0][0][-1]['type'],'recovery_feedback')
        self.assertFalse(a.seen[0][0][-1]['is_observation'])

    def test_real_entry_returns_logs_and_restores_agent_state(self):
        a=FakeAgent()
        r=run_native(a,Q,{FINISH:object(),'t':object()},lambda *a:self.fail(),observe=lambda h,t:copy.deepcopy(h),auxiliary=lambda *a:None)
        self.assertEqual(r['answer'],'Native final.')
        self.assertTrue(r['native_diagnostics'])
        self.assertIsNone(a._task_state);self.assertFalse(a._inside_state_decision)


if __name__=='__main__':unittest.main()
