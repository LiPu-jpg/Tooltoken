"""Opt-in bounded runner adapter. All model and executor access is injected.

propose(context, generate_bounded) -> {kind: tool|answer|give_up, ...}
execute(tool, arguments) -> {content: observed result, transport: status code}
validate(tool, arguments) must call the real bound tool schema validator.
No implicit simulator, judge, training or GPU entry point exists.
"""
import copy
from completion import Ledger, Budget, CompletionController, BoundedSession, deterministic_requirements, strict_json


def run(query, definitions, propose, generate, execute, validate, *, budget, max_decisions, max_validation_retries=2, answer_fits=None):
    if type(max_decisions) is not int or max_decisions<1:
        raise ValueError('explicit positive decision cap required')
    if type(max_validation_retries) is not int or max_validation_retries<0:raise ValueError('invalid validation retry cap')
    validation_failures=0
    extraction_fallback=False
    extraction_mode='provided' if definitions is not None else 'model'
    events=[]
    if definitions is None:
        extracted=generate('extract_requirements',{'original_query':query},768) if budget.reserve_generation(768) else None
        try:
            definitions=strict_json(extracted) if isinstance(extracted,str) else extracted
            ledger=Ledger(query,definitions)
        except (ValueError,TypeError,KeyError) as exc:
            ledger=Ledger(query,deterministic_requirements(query))
            ledger.coarse_fallback=True
            extraction_fallback=True
            extraction_mode='full_query_native_planner_fallback'
            events.append({'action':'requirement_extraction','raw':copy.deepcopy(extracted),
                           'accepted':False,'error_type':type(exc).__name__,'error':str(exc),
                           'fallback':extraction_mode})
        else:
            events.append({'action':'requirement_extraction','raw':copy.deepcopy(extracted),'accepted':True})
    else:
        ledger=Ledger(query,definitions)
    controller=CompletionController(ledger,budget)
    session=BoundedSession(controller,generate,answer_fits)
    history=[{'type':'user','content':query}]
    plan=None;coverage_excluded=[];coverage_recovery_active=False;force_finish=False;last_answer=''
    def finish(status,answer='',reason=''):
        return {'status':status,'answer':answer,'reason':reason,'history':history,
                'events':events+session.patch_events,'budget':vars(budget).copy(),'ledger':ledger.context(),
                'semantic_success_certified':False,'requirement_extraction_fallback':extraction_fallback,
                'requirement_extraction_mode':extraction_mode}
    def evidence_answer_or_stop(reason):
        if ledger.observations:
            route=session.preserve_answer(last_answer,reason)
            if route['answer'].strip():
                events.append({'action':'partial_answer','reason':reason,'observation_count':len(ledger.observations)})
                return finish('give_answer',route['answer'],route['reason'])
        if last_answer.strip():return finish('give_answer',last_answer,reason+'_existing_answer_preserved')
        return finish('give_up_and_restart',reason=reason+'_no_verified_patch')
    for decision_index in range(max_decisions):
        ledger.sync(history)
        context={'history':history,'completion':ledger.context(),'plan':plan,
                 'plan_is_observation':False,'recovery_excluded_tools':coverage_excluded,
                 'coverage_recovery_active':coverage_recovery_active,'force_finish':force_finish}
        proposal=propose(copy.deepcopy(context),session.generate_bounded)
        if not isinstance(proposal,dict):return finish('give_up_and_restart',reason='invalid_or_exhausted_proposal')
        kind=proposal.get('kind')
        if kind=='give_up':return evidence_answer_or_stop('policy_give_up')
        if kind=='answer':
            answer=proposal.get('answer')
            if not isinstance(answer,str) or not answer.strip():return finish('give_up_and_restart',reason='empty_answer')
            last_answer=answer.strip()
            if ledger.coarse_fallback:
                # Do not treat one coarse envelope as a completed atomic goal.
                route=session.preserve_answer(answer,'unparsed_requirements')
                events.append(route)
                return finish('give_answer',route['answer'],route['reason'])
            audit=session.generate_bounded('audit_coverage',{'answer':answer,'context':ledger.context()},768)
            try:audit=strict_json(audit) if isinstance(audit,str) else audit
            except (ValueError,TypeError):audit=None
            # A recovery needs one decision for a tool and one for the final answer.
            route=session.finish_coverage(answer,audit,allow_selection_recovery=decision_index<=max_decisions-3)
            if route['action']=='coverage_selection_recovery':
                events.append({'action':'coverage_selection_recovery',
                               'requirement_ids':route['requirement_ids'],
                               'excluded_tools':route['excluded_tools'],'is_observation':False})
                plan=generate('coverage_replan',{'requirement_ids':route['requirement_ids'],
                              'original_query':query,
                              'requirements':ledger.context()['requirements'],
                              'excluded_tools':route['excluded_tools']},route['token_limit'])
                events.append({'action':'replan','kind':'coverage','text':plan,'is_observation':False})
                if not isinstance(plan,str) or not plan.strip():
                    return finish('give_answer',answer.strip(),'coverage_replan_failed_fallback')
                coverage_excluded=route['excluded_tools'];coverage_recovery_active=True;force_finish=False
                continue
            events.append(route)
            if route['action']=='finish':return finish('give_answer',route['answer'],route['reason'])
            if route['action']=='honest_stop':return finish('give_up_and_restart',route.get('answer',''),route['reason'])
            if route['action']=='replan':
                plan=generate('replan',route,route['token_limit'])  # already reserved by controller
                events.append({'action':'replan','kind':'missing_information','text':plan,'is_observation':False})
                if not isinstance(plan,str) or not plan.strip():return finish('give_up_and_restart',answer,'replan_failed')
                continue
            raise AssertionError('unexpected completion route')
        if kind=='invalid_tool':
            tool=proposal.get('tool');goals=proposal.get('requirement_ids',[])
            if not isinstance(tool,str) or not goals or not all(isinstance(g,str) for g in goals) or not set(goals)<=set(ledger.requirements):
                return finish('give_up_and_restart',reason='invalid_recovery_proposal')
            ledger.reject(tool,proposal.get('arguments_raw',''),goals,proposal.get('reason','invalid arguments'))
            events.append({'action':'rejected_generation_arguments','tool':tool,
                           'arguments_raw':proposal.get('arguments_raw',''),
                           'reason':proposal.get('reason','invalid arguments'),'is_observation':False})
            validation_failures+=1
            if validation_failures>max_validation_retries:return finish('give_up_and_restart',reason='validation_retry_budget_exhausted')
            continue
        if kind!='tool':return finish('give_up_and_restart',reason='unknown_proposal')
        tool=proposal.get('tool');arguments=proposal.get('arguments');goals=proposal.get('requirement_ids')
        excluded=proposal.get('recovery_excluded_tools',[])
        if not isinstance(excluded,list) or not all(isinstance(identity,str) for identity in excluded) or tool in excluded:
            return finish('give_up_and_restart',reason='invalid_recovery_exclusion')
        if excluded:
            events.append({'action':'selection_recovery','excluded_tools':excluded,'is_observation':False})
        if not isinstance(tool,str) or not isinstance(goals,list) or not goals or not all(isinstance(g,str) for g in goals) or not set(goals)<=set(ledger.requirements):
            return finish('give_up_and_restart',reason='unbound_tool_goals')
        try:validate(tool,arguments)
        except (ValueError,TypeError) as exc:
            ledger.reject(tool,arguments,goals,'schema: '+str(exc))
            events.append({'action':'rejected_schema','tool':tool,'arguments':arguments,'is_observation':False})
            validation_failures+=1
            if validation_failures>max_validation_retries:return finish('give_up_and_restart',reason='validation_retry_budget_exhausted')
            # Recovery is bounded by max_decisions and the SAME model generation budget.
            continue
        permit=controller.check_call(tool,arguments,goals);events.append(permit)
        if permit['action']=='honest_stop':return finish('give_up_and_restart',reason=permit['reason'])
        if permit['action']=='reject_call':
            if controller.replans>=1:
                return evidence_answer_or_stop('duplicate_recovery_exhausted')
            if not budget.reserve_generation(96):return finish('give_up_and_restart',reason='duplicate_recovery_exhausted')
            controller.replans+=1;plan=generate('replan',ledger.context(),96)
            events.append({'action':'replan','kind':'duplicate','text':plan,'is_observation':False})
            if not isinstance(plan,str) or not plan.strip():return evidence_answer_or_stop('duplicate_replan_failed')
            continue
        history.append({'type':'call','api_identity':tool,'arguments':copy.deepcopy(arguments)})
        try:
            result=execute(tool,copy.deepcopy(arguments))
            if not isinstance(result,dict) or set(result)!={'content','transport'}:
                raise ValueError('executor must expose content and classified transport status')
        except TimeoutError:
            result={'content':{'error':'timeout'},'transport':'timeout'}
        except ConnectionResetError:
            result={'content':{'error':'connection_reset'},'transport':'connection_reset'}
        except Exception as exc:
            result={'content':{'error':type(exc).__name__,'message':str(exc)},'transport':'permanent_or_unknown'}
        history.append({'type':'observation','api_identity':tool,'content':result['content']})
        ledger.sync(history);controller.receipt(permit,result['transport'],'o'+str(len(history)-1))
        if coverage_recovery_active:
            coverage_recovery_active=False;coverage_excluded=[];force_finish=True
    if last_answer:return finish('give_answer',last_answer,'decision_budget_exhausted_answer_fallback')
    return finish('give_up_and_restart',reason='decision_budget_exhausted')
