"""CPU-only completion controller, independent of selector acceptance.

Evidence provenance is program-verified; semantic coverage is a model estimate.
No evaluation labels, model weights, network clients or training dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import copy
import hashlib
import json
import re


DIRECT_REQUEST = re.compile(
    r"\b(?:can you|could you|please|i would like(?: to)?|additionally|also|ensure|"
    r"retrieve|fetch|provide|show|display|list|find|generate|create|search|upvote|flag|give me|tell me)\b",
    re.IGNORECASE,
)
FALLBACK_REQUEST = re.compile(r"\b(?:i (?:need|want) to|we (?:need|want) to)\b", re.IGNORECASE)
ACTION_SPLIT = re.compile(
    r"\band\s+(?:then\s+)?(?=(?:retrieve|fetch|provide|show|display|list|find|generate|create|search|upvote|flag|include|give|tell)\b)",
    re.IGNORECASE,
)


def deterministic_requirements(query):
    """Lossless fallback envelope, explicitly NOT an atomic decomposition.

    Selecting only imperative sentences lost antecedents, counts and time ranges.
    The runner marks this envelope coarse and uses the original native planner.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("nonempty original query required")
    return [{"source_span": [0, len(query)], "constraints": {}}]


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('non-finite JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def span(text, bounds, maximum=240):
    if not isinstance(text, str) or not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError('text span required')
    start, end = bounds
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or end-start > maximum:
        raise ValueError('invalid or oversized span')
    return text[start:end]


@dataclass(frozen=True)
class Requirement:
    id: str
    source_span: tuple
    text: str
    constraints: dict


class Ledger:
    """Requirements bind once to original user spans; observations bind to history indices."""
    def __init__(self, query, definitions):
        if not isinstance(query, str) or not isinstance(definitions, list) or not 1 <= len(definitions) <= 12:
            raise ValueError('require original query and 1..12 atomic requirements')
        self.query = query
        self.coarse_fallback = False
        self.requirements = {}
        seen = set()
        for n, item in enumerate(definitions, 1):
            if not isinstance(item,dict) or set(item) != {'source_span', 'constraints'}:
                raise ValueError('invalid definition')
            text = span(query, item['source_span'], maximum=len(query))
            if tuple(item['source_span']) in seen:
                raise ValueError('duplicate requirement')
            seen.add(tuple(item['source_span']))
            constraints = item['constraints']
            if not isinstance(constraints, dict) or set(constraints)-{'platform','object','operation','fields','scope'}:
                raise ValueError('invalid constraints')
            # Constraints are exact user spans, not invented requirements.
            bound = {key: span(query, val, maximum=len(query)) for key, val in constraints.items()}
            rid = 'r'+str(n)
            self.requirements[rid] = Requirement(rid, tuple(item['source_span']), text, bound)
        self._history = []
        self.observations = {}
        self.rejections = []

    def sync(self, history):
        if not isinstance(history, list) or not history or history[0] != {'type':'user','content':self.query}:
            raise ValueError('original query changed')
        if history[:len(self._history)] != self._history:
            raise ValueError('history rewrite or rollback')
        self._history = copy.deepcopy(history)
        obs = {}
        for index, event in enumerate(history):
            if event.get('type') != 'observation':
                continue
            if index == 0 or history[index-1].get('type') != 'call' or history[index-1].get('api_identity') != event.get('api_identity'):
                raise ValueError('observation lacks matching executed call')
            oid = 'o'+str(index)
            obs[oid] = {'content':copy.deepcopy(event.get('content')), 'api_identity':event['api_identity'],
                        'arguments':copy.deepcopy(history[index-1]['arguments']), 'sha256':digest(event)}
        self.observations = obs

    def resolve(self, ref):
        if not isinstance(ref, dict) or set(ref)-{'observation','sha256','path','span'} or not {'observation','sha256','path'} <= set(ref):
            raise ValueError('invalid evidence reference')
        obs = self.observations.get(ref['observation'])
        if obs is None or ref['sha256'] != obs['sha256']:
            raise ValueError('missing or changed observation')
        value = obs['content']
        if isinstance(value, dict) and value.get('error'):
            raise ValueError('failed execution is not support')
        path = ref['path']
        if not isinstance(path, list) or len(path)>12:
            raise ValueError('bounded typed field path required')
        for key in path:
            # Explicit decoder hop; no Python literal eval or implicit coercion.
            if isinstance(key, dict) and key == {'decode':'json'}:
                if not isinstance(value,str):raise ValueError('decode requires JSON text')
                value = strict_json(value)
            elif isinstance(value, dict) and isinstance(key,str) and key in value:
                value=value[key]
            elif isinstance(value,list) and type(key) is int and 0<=key<len(value):
                value=value[key]
            else:
                raise ValueError('absent field path')
        if 'span' in ref:
            value=span(value,ref['span'])
        if value is None or value=='' or value==[] or value=={} or len(compact(value))>4096:
            raise ValueError('empty or unbounded evidence; narrow path or span')
        return copy.deepcopy(value)

    def reject(self, tool, arguments, requirement_ids, reason, executed=False):
        if not requirement_ids or not set(requirement_ids)<=set(self.requirements) or not reason:
            raise ValueError('rejection must retain atomic goals and reason')
        self.rejections.append({'tool':tool,'arguments':copy.deepcopy(arguments),'requirement_ids':list(requirement_ids),
                                'reason':reason[:300],'executed':bool(executed),'is_observation':False})

    def context(self):
        return {'original_query':self.query, 'requirements_are_atomic':not self.coarse_fallback,
                'requirements':[copy.deepcopy(vars(r)) for r in self.requirements.values()],
                'observations':copy.deepcopy(self.observations),
                'rejected_proposals':copy.deepcopy(self.rejections),
                'warning':'Only observations are evidence. Coverage and constraint matches are estimates, not program-certified entailment.'}

    def validate_audit(self, audit, answer):
        if not isinstance(audit,dict) or set(audit)!={'requirements'} or not isinstance(audit['requirements'],list):
            raise ValueError('invalid audit')
        result={}
        for row in audit['requirements']:
            if not isinstance(row,dict) or set(row)!={'id','status','evidence','constraint_matches','answer_spans','note'}:
                raise ValueError('invalid audit fields')
            rid=row['id']
            if not isinstance(rid,str) or rid not in self.requirements or rid in result or not isinstance(row['status'],str) or row['status'] not in {'pending','supported','blocked'}:
                raise ValueError('unknown/duplicate requirement or status')
            if not isinstance(row['note'],str) or len(row['note'])>160:
                raise ValueError('bounded note required')
            refs=row['evidence']
            if not isinstance(refs,list) or len(refs)>4:
                raise ValueError('bounded evidence required')
            resolved=[self.resolve(ref) for ref in refs]
            matches=row['constraint_matches']
            if not isinstance(matches,dict) or set(matches)!=set(self.requirements[rid].constraints) or any(type(v) is not bool for v in matches.values()):
                raise ValueError('all original constraints must be explicitly assessed')
            if row['status']=='supported' and (not resolved or not all(matches.values())):
                raise ValueError('support needs evidence and estimated constraint match')
            answer_spans=row['answer_spans']
            if not isinstance(answer_spans,list) or len(answer_spans)>4:
                raise ValueError('bounded answer spans required')
            for bounds in answer_spans:span(answer,bounds,maximum=1000)
            if row['status']!='supported' and answer_spans:
                raise ValueError('unsupported requirement cannot be marked answered')
            result[rid]=copy.deepcopy(row)
        if set(result)!=set(self.requirements):
            raise ValueError('audit must retain every stable requirement ID')
        return result

    def validate_coverage_audit(self, audit):
        """Validate routing shape only; model coverage claims remain untrusted."""
        if not isinstance(audit, dict) or set(audit) != {'requirements'} or not isinstance(audit['requirements'], list):
            raise ValueError('invalid coverage audit')
        result = {}
        for row in audit['requirements']:
            if not isinstance(row, dict) or set(row) != {'id', 'observation_status', 'answer_status'}:
                raise ValueError('invalid coverage row')
            rid = row['id']
            if not isinstance(rid, str) or rid not in self.requirements or rid in result:
                raise ValueError('unknown or duplicate coverage ID')
            if row['observation_status'] not in {'present', 'missing', 'blocked'}:
                raise ValueError('invalid observation status')
            if row['answer_status'] not in {'expressed', 'omitted'}:
                raise ValueError('invalid answer status')
            result[rid] = copy.deepcopy(row)
        if set(result) != set(self.requirements):
            raise ValueError('coverage audit must retain every stable requirement ID')
        return result


@dataclass
class Budget:
    max_calls: int
    max_generations: int
    max_tokens: int
    calls: int = 0
    generations: int = 0
    tokens: int = 0

    def __post_init__(self):
        if any(type(v) is not int or v<0 for v in vars(self).values()):raise ValueError('nonnegative integer budgets required')

    def reserve_generation(self, token_limit):
        if type(token_limit) is not int or token_limit<1:raise ValueError('positive token reservation required')
        if self.generations>=self.max_generations or self.tokens+token_limit>self.max_tokens:return False
        # Full requested capacity is charged. No hidden or refunded recovery tokens.
        self.generations+=1;self.tokens+=token_limit
        return True

    def reserve_call(self):
        if self.calls>=self.max_calls:return False
        self.calls+=1;return True


class CompletionController:
    def __init__(self, ledger, budget):
        self.ledger=ledger;self.budget=budget
        self.answer_repairs=0;self.replans=0;self.audit_errors=0;self.coverage_recoveries=0
        self.executed=[];self.transient_retries={};self.pending={}

    def coverage_route(self, answer, audit, allow_selection_recovery=True):
        if self.ledger.coarse_fallback:
            return {'action':'finish','answer':answer,'reason':'unparsed_requirements_no_coverage_claim'}
        if audit is None:
            return {'action':'finish','answer':answer,'reason':'coverage_audit_unavailable_fallback'}
        try:
            rows=self.ledger.validate_coverage_audit(audit)
        except (ValueError,TypeError,KeyError):
            self.audit_errors+=1
            return {'action':'finish','answer':answer,'reason':'invalid_coverage_audit_fallback'}
        missing=[rid for rid,row in rows.items() if row['observation_status']=='missing']
        omitted=[rid for rid,row in rows.items()
                 if row['observation_status']=='present' and row['answer_status']=='omitted']
        blocked=[rid for rid,row in rows.items() if row['observation_status']=='blocked']
        if (missing and allow_selection_recovery and self.coverage_recoveries<1
                and self.budget.calls<self.budget.max_calls and self.budget.reserve_generation(128)):
            self.coverage_recoveries+=1
            excluded=sorted({row['tool'] for row in self.ledger.rejections
                             if isinstance(row.get('tool'),str) and row.get('reason')})
            return {'action':'coverage_selection_recovery','requirement_ids':missing,
                    'excluded_tools':excluded,'original_answer':answer,
                    'context':self.ledger.context(),'token_limit':128,'is_observation':False}
        if omitted and self.answer_repairs<1 and self.budget.reserve_generation(1024):
            self.answer_repairs+=1
            return {'action':'coverage_answer_repair','requirement_ids':omitted,
                    'original_answer':answer,'audit':rows,'token_limit':1024,'is_observation':False}
        reason='coverage_estimate_complete' if not missing and not omitted and not blocked else 'bounded_coverage_recovery_complete'
        return {'action':'finish','answer':answer,'reason':reason,
                'unresolved':missing+omitted+blocked}

    def finish_route(self, answer, audit):
        if audit is None:return {'action':'finish','answer':answer,'reason':'audit_unavailable_fallback'}
        try:rows=self.ledger.validate_audit(audit,answer)
        except (ValueError,TypeError,KeyError):
            self.audit_errors+=1
            return {'action':'finish','answer':answer,'reason':'invalid_audit_fallback'}
        missing=[k for k,r in rows.items() if r['status']!='supported']
        omitted=[k for k,r in rows.items() if r['status']=='supported' and not r['answer_spans']]
        if not missing and not omitted:return {'action':'finish','answer':answer,'reason':'estimated_coverage_complete'}
        # Already acquired information is an answer revision problem, even if other goals remain blocked.
        if omitted and self.answer_repairs<1 and self.budget.reserve_generation(1024):
            self.answer_repairs+=1
            return {'action':'repair_answer','requirement_ids':omitted,'missing_information':missing,'original_answer':answer,'audit':rows,'token_limit':1024}
        pending=[k for k in missing if rows[k]['status']=='pending']
        if pending and self.replans<1 and self.budget.calls<self.budget.max_calls and self.budget.reserve_generation(96):
            self.replans+=1
            return {'action':'replan','requirement_ids':pending,'original_answer':answer,'context':self.ledger.context(),'token_limit':96}
        return {'action':'honest_stop','answer':answer,'unresolved':missing+omitted,'reason':'bounded_recovery_exhausted_or_blocked'}

    def check_call(self, tool, arguments, requirement_ids):
        """Only an execution receipt creates retry state; proposals never do."""
        if not isinstance(arguments,dict):raise ValueError('arguments must be an object, including legal {}')
        key=digest([tool,arguments]);prior=[r for r in self.executed if r['key']==key]
        if prior:
            last=prior[-1]
            new_information=any(oid not in last['seen'] and digest([o['api_identity'],o['arguments']])!=key
                                for oid,o in self.ledger.observations.items())
            if not new_information:
                retryable=last['transport'] in {'timeout','connection_reset','http_429','http_502','http_503','http_504'}
                if retryable and self.transient_retries.get(key,0)<1:
                    self.transient_retries[key]=self.transient_retries.get(key,0)+1
                else:
                    self.ledger.reject(tool,arguments,requirement_ids,'identical call without independent new information',executed=False)
                    return {'action':'reject_call','reason':'no_new_information','key':key}
        if not self.budget.reserve_call():return {'action':'honest_stop','reason':'call_budget_exhausted'}
        permit={'action':'execute','key':key,'tool':tool,'arguments':copy.deepcopy(arguments),'serial':self.budget.calls}
        self.pending[permit['serial']]={'permit':copy.deepcopy(permit),'observations':set(self.ledger.observations)}
        return permit

    def receipt(self, permit, transport, observation_id):
        pending=self.pending.get(permit.get('serial'))
        if permit.get('action')!='execute' or pending is None or pending['permit']!=permit:
            raise ValueError('unexecuted or altered proposal cannot create receipt')
        obs=self.ledger.observations.get(observation_id)
        if obs is None or observation_id in pending['observations'] or digest([obs['api_identity'],obs['arguments']])!=permit['key']:
            raise ValueError('receipt must bind a real matching observation')
        del self.pending[permit['serial']]
        self.executed.append({'key':permit['key'],'transport':transport,'seen':set(self.ledger.observations)})


class BoundedSession:
    """Adapter for a future runner. Every generator uses the SAME Budget object.

    This CPU implementation deliberately requires explicit callbacks; it cannot
    accidentally load a model, call MirrorAPI, judge, or launch a new experiment.
    """
    def __init__(self, controller, generate, answer_fits=None):
        self.controller=controller;self.generate=generate
        self.patch_events=[]
        self.answer_fits=answer_fits

    def preserve_answer(self, answer, reason):
        from evidence_patch import repair
        if self.controller.answer_repairs >= 1:
            return {'action':'finish','answer':answer,'reason':reason+'_patch_already_attempted'}
        self.controller.answer_repairs += 1
        revised, event = repair(answer, self.controller.ledger, self.generate_bounded, self.answer_fits)
        self.patch_events.append(event)
        return {'action':'finish','answer':revised,'reason':reason+'_'+event['outcome']}

    def generate_bounded(self, purpose, payload, token_limit):
        if not self.controller.budget.reserve_generation(token_limit):return None
        return self.generate(purpose,copy.deepcopy(payload),token_limit)

    def finish(self, answer, audit):
        route=self.controller.finish_route(answer,audit)
        if route['action']=='repair_answer':
            from evidence_patch import repair
            # Selection capacity was already charged by finish_route. Review
            # still uses the shared budget; it can never run for free.
            def bounded(purpose,payload,limit):
                return (self.generate(purpose,payload,limit) if purpose=='select_evidence_patch'
                        else self.generate_bounded(purpose,payload,limit))
            revised,event=repair(answer,self.controller.ledger,bounded,self.answer_fits)
            self.patch_events.append(event)
            return {'action':'finish','answer':revised,'reason':'answer_'+event['outcome']}
        return route

    def finish_coverage(self, answer, audit, *, allow_selection_recovery=True):
        route=self.controller.coverage_route(answer,audit,allow_selection_recovery)
        if route['action']=='coverage_answer_repair':
            from evidence_patch import repair
            def bounded(purpose,payload,limit):
                return (self.generate(purpose,payload,limit) if purpose=='select_evidence_patch'
                        else self.generate_bounded(purpose,payload,limit))
            revised,event=repair(answer,self.controller.ledger,bounded,self.answer_fits)
            self.patch_events.append(event)
            return {'action':'finish','answer':revised,'reason':'coverage_'+event['outcome']}
        return route
