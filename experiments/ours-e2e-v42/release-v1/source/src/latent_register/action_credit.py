"""Verifiable local action constraints, separate from trajectory outcome credit.

Do not infer correctness from JSON validity. Correct JSON may still select the wrong
entity or tool. No synthetic correct values or reference answers enter this audit.
"""
import copy
from .toolbench_data import compact, strict_json


def classify_arguments(event, tools):
    if event['kind'] not in {'arguments', 'repair'}:
        return None
    if event['kind'] == 'repair' and event.get('stopped') and event.get('text', '').strip() == '<reselect>':
        return {'valid': None, 'reason': 'reselect_control'}
    if not event.get('stopped'):
        return {'valid': False, 'reason': 'incomplete_arguments'}
    if event['selected'] not in tools:
        raise ValueError('Unbound tool in action audit')
    try:
        arguments = strict_json(event['text'])
    except ValueError:
        return {'valid': False, 'reason': 'invalid_json'}
    try:
        tools[event['selected']].validate_arguments(arguments)
    except ValueError as exc:
        return {'valid': False, 'reason': 'invalid_schema', 'validation_error': str(exc)}
    return {'valid': True, 'reason': 'schema_valid_only'}


def local_credit(event, trajectory_advantage, tools, *, negative_floor=.25, terminal_complete=None):
    if not 0 < negative_floor <= 1:
        raise ValueError('Bounded negative floor required')
    result = copy.deepcopy(event)
    audit = classify_arguments(event, tools)
    tags = []
    if audit and audit['valid'] is False:
        tags.append('invalid_argument_action')
    # Set by runtime only for repeated successful content under an explicit
    # stable-read contract. A fresh-data request or unclassified tool is exempt.
    if event.get('redundancy_evidence', {}).get('certified') is True:
        evidence = event['redundancy_evidence']
        if (event['kind'] not in {'arguments', 'repair'} or not audit or audit['valid'] is not True
            or evidence.get('stable_read_contract') is not True or evidence.get('same_successful_content') is not True
            or evidence.get('freshness_requested') is not False):
            raise ValueError('Incomplete redundant-call certification')
        tags.append('redundant_successful_call')
    effective = min(float(trajectory_advantage), -negative_floor) if tags else float(trajectory_advantage)
    if terminal_complete is not None and type(terminal_complete) is not bool:
        raise ValueError('Terminal completion must be boolean or unavailable')
    # Partial progress may teach useful upstream actions/answer content. It does
    # not certify that stopping now was correct. This is a conservative positive
    # credit cap, not a deterministic failure penalty or a replacement for judge
    # calibration. Preserve negative credit and never turn refusal into success.
    if (terminal_complete is False and event['kind']=='control'
        and event.get('stopped') is True and event.get('text','').strip()=='<final>'):
        tags.append('incomplete_terminal_positive_credit_cap')
        effective=min(effective,0.)
    result.update(trajectory_advantage=float(trajectory_advantage), advantage=effective,
                  local_credit={'version': 'v30', 'tags': tags, 'argument_audit': audit,
                                'negative_floor': negative_floor,
                                'terminal_complete': terminal_complete,
                                'scope': 'own-action surrogate sign; not a guarantee about shared-parameter interference'})
    return result


def certify_repeat(previous, current, *, stable_read_contract, freshness_requested):
    """Explicit caller/data contract: never classify unknown or fresh-data tools."""
    certified = bool(stable_read_contract is True and freshness_requested is False
                     and previous is not None
                     and current['api_identity'] == previous['api_identity']
                     and compact(current['arguments']) == compact(previous['arguments'])
                     and current['result'].get('error') == previous['result'].get('error') == ''
                     and compact(current['result']['response']) == compact(previous['result']['response']))
    return {'certified': certified, 'stable_read_contract': stable_read_contract,
            'freshness_requested': freshness_requested, 'same_successful_content': certified}


def validate_repeat_contracts(contracts, tasks, tools):
    import hashlib
    queries={t['id']:t['query'] for t in tasks}
    if set(contracts)-set(queries):raise ValueError('Unknown task repeat contract')
    for task, rules in contracts.items():
        for identity, rule in rules.items():
            if identity not in tools:raise ValueError('Unknown repeat-contract API')
            if rule.get('source_query_sha256') != hashlib.sha256(queries[task].encode()).hexdigest():
                raise ValueError('Repeat-contract query changed')
            if rule.get('document_sha256') != hashlib.sha256(tools[identity].registration_document.encode()).hexdigest():
                raise ValueError('Repeat-contract document changed')
            if rule.get('stable_read_contract') is not True or rule.get('freshness_requested') is not False or not rule.get('review_basis'):
                raise ValueError('Repeat-contract review missing')
