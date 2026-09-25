"""Bounded, source-exact answer edits, never an unconstrained rewrite.

The model selects literal quotes and optional paragraph replacements. Programs
verify provenance and render values; a separate model review estimates relevance
and preservation. That review is NOT a semantic correctness certificate.
"""
from __future__ import annotations
import copy
import re
from completion import compact, digest, strict_json

MAX_SOURCE_CHARS = 8192
MAX_CATALOG_CHARS = 24000
MAX_QUOTES = 4
MAX_QUOTE_CHARS = 1200
MAX_ADDED_CHARS = 4096


def paragraphs(answer):
    # Spans are computed by Python. The model uses p1, p2, ...; never offsets.
    out = {}
    for match in re.finditer(r'\S(?:.*?\S)?(?=\n\s*\n|\Z)', answer, re.S):
        key = 'p' + str(len(out) + 1)
        out[key] = {'text':match.group(), 'span':list(match.span())}
    return out


def catalogue(ledger):
    cards = {}; omitted = []; used = 0
    # Latest results first. Never silently truncate a response into a false fact.
    for oid, obs in reversed(list(ledger.observations.items())):
        content = obs['content']
        value = content.get('response', content) if isinstance(content,dict) else content
        text = value if isinstance(value,str) else compact(value)
        # Empty arrays/objects/zero/false ARE actual results, not execution errors.
        if text == '':
            text = compact(content)
        card = {'observation':oid, 'api_identity':obs['api_identity'],
                'arguments':copy.deepcopy(obs['arguments']), 'source_text':text,
                'observation_sha256':obs['sha256'],
                'error_envelope':copy.deepcopy(content.get('error')) if isinstance(content,dict) else None,
                'warning':'A response, including an empty result, does not certify task success.'}
        size = len(compact(card))
        reason = ('binary_or_control_characters' if any(ord(c)<32 and c not in '\n\r\t' for c in text)
                  else 'source_too_long' if len(text)>MAX_SOURCE_CHARS
                  else 'catalog_budget' if used+size>MAX_CATALOG_CHARS else None)
        if reason:
            omitted.append({'observation':oid,'reason':reason,'characters':len(text)})
        else:
            cards[oid] = card; used += size
    return cards, omitted


def apply_plan(answer, plan, cards):
    if not isinstance(plan,dict) or set(plan)!={'append','replace'}:
        raise ValueError('require append and replace arrays only')
    if not all(isinstance(plan[k],list) for k in plan):
        raise ValueError('patch arrays required')
    if len(plan['append'])+len(plan['replace'])>MAX_QUOTES:
        raise ValueError('too many evidence quotes')
    units=paragraphs(answer); edits=[]; seen=set(); refs=[]
    def quote(row, replacement=False):
        keys={'observation','quote'}|({'paragraph'} if replacement else set())
        if not isinstance(row,dict) or set(row)!=keys:
            raise ValueError('invalid evidence edit')
        oid=row['observation']; text=row['quote']
        if not isinstance(oid,str) or oid not in cards or not isinstance(text,str) or not text.strip():
            raise ValueError('unknown observation or empty quote')
        if len(text)>MAX_QUOTE_CHARS or text not in cards[oid]['source_text']:
            raise ValueError('quote absent from exact source or too long')
        # Quotes are inserted literally: no labels/values supplied by a rewriter.
        refs.append({'observation':oid,'quote':text,'observation_sha256':cards[oid]['observation_sha256']})
        return text
    for row in plan['replace']:
        text=quote(row,True); key=row['paragraph']
        if not isinstance(key,str) or key not in units or key in seen:
            raise ValueError('unknown or repeated original paragraph')
        seen.add(key); edits.append((*units[key]['span'],text))
    revised=answer
    for start,end,text in sorted(edits,reverse=True):
        revised=revised[:start]+text+revised[end:]
    for row in plan['append']:
        text=quote(row)
        if text not in revised:
            revised += ('\n\n' if revised.strip() else '')+text
    if sum(len(r['quote']) for r in refs)>MAX_ADDED_CHARS:
        raise ValueError('added content budget exceeded')
    return revised,refs


def repair(answer, ledger, generate_bounded, answer_fits=None):
    cards,omitted=catalogue(ledger)
    event={'action':'evidence_patch','answer_before':answer,'answer_after':answer,
           'original_query_sha256':digest(ledger.query),'omitted_sources':omitted,
           'source_ids':list(cards),'semantic_success_certified':False}
    def unchanged(reason):
        event['outcome']=reason
        return answer,event
    if not cards:return unchanged('no_eligible_source')
    payload={'original_query':ledger.query,'answer':answer,
             'paragraphs':{k:v['text'] for k,v in paragraphs(answer).items()},
             'sources':list(cards.values()),'omitted_sources':omitted}
    raw=generate_bounded('select_evidence_patch',payload,1024)
    event['raw_plan']=copy.deepcopy(raw)
    try:
        plan=strict_json(raw) if isinstance(raw,str) else raw
        candidate,refs=apply_plan(answer,plan,cards)
    except (ValueError,TypeError,KeyError) as exc:
        event['error']=str(exc);return unchanged('invalid_patch_preserved')
    event.update(plan=plan,candidate=candidate,evidence=refs)
    if candidate==answer:return unchanged('no_change')
    if answer_fits is not None and not answer_fits(candidate):
        return unchanged('answer_budget_preserved')
    review_raw=generate_bounded('review_evidence_patch',dict(payload,candidate=candidate,patch=plan),768)
    event['raw_review']=copy.deepcopy(review_raw)
    try:
        review=strict_json(review_raw) if isinstance(review_raw,str) else review_raw
        required={'matches_request','preserves_supported','no_contradiction'}
        if not isinstance(review,dict) or set(review)!=required or any(type(review[k]) is not bool for k in required):
            raise ValueError('review must contain all three explicit booleans')
        if not all(review.values()):return unchanged('review_rejected_preserved')
        if any(ledger.observations[r['observation']]['sha256']!=r['observation_sha256'] for r in refs):
            raise ValueError('source changed after selection')
    except (ValueError,TypeError,KeyError) as exc:
        event['error']=str(exc);return unchanged('review_unavailable_preserved')
    event.update(outcome='source_patch_applied',answer_after=candidate,
                 provenance_checked=True,semantic_review='model_estimate')
    return candidate,event
