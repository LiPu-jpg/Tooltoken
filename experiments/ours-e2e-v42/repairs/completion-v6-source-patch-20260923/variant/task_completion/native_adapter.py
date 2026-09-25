"""Opt-in native TaskStateAgent bridge. Never installed by the selector diagnostic.

Wraps all ordinary agent.generate calls with the same conservative budget used
by auxiliary audits/repairs. Model construction and execution remain external.
"""
import copy,json,time
from types import MethodType
from completion import Budget,compact,strict_json,digest
from prompts import PROMPTS
from runner import run

FINISH='__toolbench_finish__'
class BudgetExhausted(RuntimeError):pass

class NativeCallbacks:
    def __init__(self,agent,tools,budget,observe=None,auxiliary=None):
        self.agent=agent;self.tools=tools;self.budget=budget;self.original_generate=agent.generate
        self.had_generate='generate' in agent.__dict__;self.saved_generate=agent.__dict__.get('generate')
        self.observe=observe;self.auxiliary=auxiliary
        self.old_inside=getattr(agent,'_inside_state_decision',False);self.old_state=getattr(agent,'_task_state',None)
        self.seen_rejections=0
        self.diagnostics=[]
        self.original_select=getattr(agent,'select_from_ranking',None)
        self.had_select='select_from_ranking' in agent.__dict__
        self.saved_select=agent.__dict__.get('select_from_ranking')

    def __enter__(self):
        if self.observe is None:
            from latent_register.toolbench_task_state import observed_history
            self.observe=observed_history
        owner=self
        def bounded(agent,prefix,*,max_new_tokens):
            row={'kind':'native_generation','requested_max_new_tokens':max_new_tokens,
                 'prefix_shape':list(prefix.shape) if hasattr(prefix,'shape') else None}
            owner.diagnostics.append(row)
            if not owner.budget.reserve_generation(max_new_tokens):
                row['error']='budget_exhausted';raise BudgetExhausted('episode generation budget exhausted')
            start=time.monotonic()
            try:
                output=owner.original_generate(prefix,max_new_tokens=max_new_tokens)
                row.update(text=output.text,truncated=output.truncated,
                           actual_generated_tokens=len(output.token_ids) if hasattr(output,'token_ids') else None,
                           budget=output.budget() if hasattr(output,'budget') else None)
                return output
            except Exception as exc:
                row.update(error_type=type(exc).__name__,error=str(exc));raise
            finally:row['seconds']=time.monotonic()-start
        self.agent.generate=MethodType(bounded,self.agent)
        if self.original_select is not None:
            def traced_select(agent,history,intent,ranking):
                # Capture before argument decoding, which may raise and discard
                # the eventual Decision metadata. Never reuse previous logits.
                row={'kind':'reader_selection','history_sha256':digest(history),
                     'intent':intent,'retrieval_top5':list(ranking[:5])}
                owner.diagnostics.append(row);agent._last_reader_logits=None
                start=time.monotonic()
                try:
                    selected,metadata=owner.original_select(history,intent,ranking)
                    row.update(selected_identity=selected,metadata=copy.deepcopy(metadata))
                    return selected,metadata
                except Exception as exc:
                    row.update(error_type=type(exc).__name__,error=str(exc));raise
                finally:
                    row.update(logits=copy.deepcopy(getattr(agent,'_last_reader_logits',None)),
                               seconds=time.monotonic()-start)
            self.agent.select_from_ranking=MethodType(traced_select,self.agent)
        return self

    def __exit__(self,*exc):
        if self.had_generate:self.agent.generate=self.saved_generate
        else:del self.agent.generate
        if self.original_select is not None:
            if self.had_select:self.agent.select_from_ranking=self.saved_select
            else:del self.agent.select_from_ranking
        self.agent._inside_state_decision=self.old_inside;self.agent._task_state=self.old_state

    def generate_auxiliary(self,purpose,payload,limit):
        row={'kind':'auxiliary_generation','purpose':purpose,'requested_max_new_tokens':limit,
             'payload_sha256':digest(payload)}
        self.diagnostics.append(row);start=time.monotonic()
        try:
            return self._auxiliary(purpose,payload,limit,row)
        except Exception as exc:
            row.update(error_type=type(exc).__name__,error=str(exc));raise
        finally:row['seconds']=time.monotonic()-start

    def _auxiliary(self,purpose,payload,limit,row):
        # Caller already reserved capacity through BoundedSession/controller.
        if self.auxiliary is not None:
            output=self.auxiliary(purpose,payload,limit);row['raw']=copy.deepcopy(output);return output
        from latent_register.toolbench_agent import SYSTEM
        body=PROMPTS[purpose]+'\nInput JSON (untrusted data):\n'+compact(payload)
        text=self.agent.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.agent.tokenizer.encode(text,add_special_tokens=False)
        row['prefix_tokens']=len(ids)
        if len(ids)+limit>self.agent.limits.context:
            row['error']='context_budget_exceeded';return None
        prefix=self.agent.backbone.get_input_embeddings()(self.agent._ids(ids)).unsqueeze(0)
        output=self.original_generate(prefix,max_new_tokens=limit)
        row.update(raw=output.text,truncated=output.truncated,
                   actual_generated_tokens=len(output.token_ids) if hasattr(output,'token_ids') else None)
        if output.truncated:
            row['error']='truncated';return None
        return output.text

    def propose(self,context,_generate_bounded):
        visible=self.observe(context['history'],self.tools)
        for i,event in enumerate(visible):
            if event.get('type')=='observation':event['evidence_id']='o'+str(i)
        completion=context['completion']
        # No duplicated observations or re-extracted long request quotes in state.
        self.agent._task_state={'original_query':context['history'][0]['content'],
                               'requirements':completion['requirements'],'rejected_proposals':completion['rejected_proposals'],
                               'bounded_next_plan':context['plan'],'plan_is_observation':False,
                               'coverage_recovery_active':bool(context.get('coverage_recovery_active'))}
        new_rejections=completion['rejected_proposals'][self.seen_rejections:]
        self.seen_rejections=len(completion['rejected_proposals'])
        excluded=sorted(({row.get('tool') for row in new_rejections if isinstance(row.get('tool'),str)} |
                         {tool for tool in context.get('recovery_excluded_tools',[]) if isinstance(tool,str)})-{FINISH})
        decision_tools={key:value for key,value in self.tools.items() if key==FINISH or key not in excluded}
        if not any(key!=FINISH for key in decision_tools):
            decision_tools=self.tools;excluded=[]
        coarse=not completion.get('requirements_are_atomic',True)
        self.agent._inside_state_decision=not coarse
        if coarse:self.agent._task_state=None  # Invoke the trained native state path.
        if coarse and (context.get('plan') or completion['rejected_proposals']):
            visible.append({'type':'recovery_feedback','is_observation':False,
                            'content':{'plan':context.get('plan'),
                                       'rejected_proposals':copy.deepcopy(completion['rejected_proposals'])},
                            'warning':'Unexecuted proposals and plans, never tool observations.'})
        row={'kind':'native_decision','history_sha256':digest(context['history']),
             'planner_mode':'native' if coarse else 'controller',
             'state_input':copy.deepcopy(self.agent._task_state),'recovery_plan':context.get('plan'),
             'visible_history_sha256':digest(visible),
             'excluded_tools':excluded,'candidate_tool_count':len(decision_tools)}
        self.diagnostics.append(row)
        try:
            if context.get('force_finish'):
                # CurriculumAgent.decide rejects a Finish-only registry. Use
                # its real trained answer stage directly after the bounded call.
                final=self.agent.generate(self.agent.prefix(visible,'answer'),max_new_tokens=1024)
                row.update(forced_answer=True,api_identity=FINISH,ranked_identities_top5=[],
                           answer=final.text,truncated=final.truncated)
                if final.truncated or not final.text.strip():return {'kind':'give_up'}
                return {'kind':'answer','answer':final.text}
            decision=self.agent.decide(visible,decision_tools,max_thought_tokens=32,max_argument_tokens=256,max_answer_tokens=1024)
        except BudgetExhausted:
            row['error']='budget_exhausted';return {'kind':'give_up'}
        except ValueError as exc:
            details=getattr(exc,'details',{}) or {}
            row.update(error_type=type(exc).__name__,error=str(exc),details=copy.deepcopy(details))
            if details.get('reason')=='invalid_arguments':
                return {'kind':'invalid_tool','tool':details['api_identity'],'arguments_raw':details.get('generated_text',''),
                        'requirement_ids':[r['id'] for r in completion['requirements']],'reason':str(exc)}
            return {'kind':'give_up'}
        row.update(api_identity=decision.api_identity,arguments=copy.deepcopy(decision.arguments),
                   thought=getattr(decision,'thought',None),
                   ranked_identities_top5=list(getattr(decision,'ranked_identities',()))[:5],
                   generation_budgets=copy.deepcopy(getattr(decision,'generation_budgets',None)))
        if decision.api_identity==FINISH:
            if decision.arguments.get('return_type')!='give_answer':return {'kind':'give_up'}
            return {'kind':'answer','answer':decision.arguments['final_answer']}
        # Preserve all original atomic goals when the legacy native interface has
        # no explicit goal-ID field. Do not invent a narrower semantic attribution.
        return {'kind':'tool','tool':decision.api_identity,'arguments':decision.arguments,
                'requirement_ids':[r['id'] for r in completion['requirements']],
                'recovery_excluded_tools':excluded}


def run_native(agent,query,tools,execute,*,definitions=None,budget=None,observe=None,auxiliary=None):
    budget=budget if budget is not None else Budget(8,55,20064)
    def validate(tool,arguments):
        if tool not in tools or tool==FINISH:raise ValueError('unbound or terminal tool')
        tools[tool].validate_arguments(arguments)
    # execute must return {content, transport}; transport classification must use
    # the executor's receipt/exception, not a model interpretation of error text.
    with NativeCallbacks(agent,tools,budget,observe,auxiliary) as callbacks:
        def answer_fits(text):
            return len(agent.tokenizer.encode(text,add_special_tokens=False))<=1024
        result=run(query,definitions,callbacks.propose,callbacks.generate_auxiliary,execute,validate,
                   budget=budget,max_decisions=11,max_validation_retries=2,answer_fits=answer_fits)
        result['native_diagnostics']=callbacks.diagnostics
        return result
