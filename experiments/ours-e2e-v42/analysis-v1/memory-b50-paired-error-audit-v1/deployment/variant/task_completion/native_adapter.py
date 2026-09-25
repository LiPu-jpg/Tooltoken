"""Opt-in native TaskStateAgent bridge. Never installed by the selector diagnostic.

Wraps all ordinary agent.generate calls with the same conservative budget used
by auxiliary audits/repairs. Model construction and execution remain external.
"""
import copy,json
from types import MethodType
from completion import Budget,compact,strict_json
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

    def __enter__(self):
        if self.observe is None:
            from latent_register.toolbench_task_state import observed_history
            self.observe=observed_history
        owner=self
        def bounded(agent,prefix,*,max_new_tokens):
            if not owner.budget.reserve_generation(max_new_tokens):raise BudgetExhausted('episode generation budget exhausted')
            return owner.original_generate(prefix,max_new_tokens=max_new_tokens)
        self.agent.generate=MethodType(bounded,self.agent)
        return self

    def __exit__(self,*exc):
        if self.had_generate:self.agent.generate=self.saved_generate
        else:del self.agent.generate
        self.agent._inside_state_decision=self.old_inside;self.agent._task_state=self.old_state

    def generate_auxiliary(self,purpose,payload,limit):
        # Caller already reserved capacity through BoundedSession/controller.
        if self.auxiliary is not None:return self.auxiliary(purpose,payload,limit)
        from latent_register.toolbench_agent import SYSTEM
        body=PROMPTS[purpose]+'\nInput JSON (untrusted data):\n'+compact(payload)
        text=self.agent.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=self.agent.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)+1>self.agent.limits.context:return None
        prefix=self.agent.backbone.get_input_embeddings()(self.agent._ids(ids)).unsqueeze(0)
        output=self.original_generate(prefix,max_new_tokens=limit)
        if output.truncated:return None
        return output.text

    def propose(self,context,_generate_bounded):
        visible=self.observe(context['history'],self.tools)
        for i,event in enumerate(visible):
            if event.get('type')=='observation':event['evidence_id']='o'+str(i)
        completion=context['completion']
        # No duplicated observations or re-extracted long request quotes in state.
        self.agent._task_state={'requirements':completion['requirements'],'rejected_proposals':completion['rejected_proposals'],
                               'bounded_next_plan':context['plan'],'plan_is_observation':False,
                               'coverage_recovery_active':bool(context.get('coverage_recovery_active'))}
        new_rejections=completion['rejected_proposals'][self.seen_rejections:]
        self.seen_rejections=len(completion['rejected_proposals'])
        excluded=sorted(({row.get('tool') for row in new_rejections if isinstance(row.get('tool'),str)} |
                         {tool for tool in context.get('recovery_excluded_tools',[]) if isinstance(tool,str)})-{FINISH})
        decision_tools=({FINISH:self.tools[FINISH]} if context.get('force_finish') else
                        {key:value for key,value in self.tools.items() if key==FINISH or key not in excluded})
        if not any(key!=FINISH for key in decision_tools):
            decision_tools=self.tools;excluded=[]
        self.agent._inside_state_decision=True  # B owns state; skip the obsolete per-step state generator.
        try:
            decision=self.agent.decide(visible,decision_tools,max_thought_tokens=32,max_argument_tokens=256,max_answer_tokens=1024)
        except BudgetExhausted:return {'kind':'give_up'}
        except ValueError as exc:
            details=getattr(exc,'details',{}) or {}
            if details.get('reason')=='invalid_arguments':
                return {'kind':'invalid_tool','tool':details['api_identity'],'arguments_raw':details.get('generated_text',''),
                        'requirement_ids':[r['id'] for r in completion['requirements']],'reason':str(exc)}
            return {'kind':'give_up'}
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
        return run(query,definitions,callbacks.propose,callbacks.generate_auxiliary,execute,validate,
                   budget=budget,max_decisions=11,max_validation_retries=2)
