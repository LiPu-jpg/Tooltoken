"""Reviewed TRAIN-only, record-specific hard candidates; no model changes."""
import hashlib
import json
import math
import random
from pathlib import Path


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def record_key(record):
    return stable(record['id'])


def fingerprint(record):
    return hashlib.sha256(stable(record).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_reviewed(path, prepared, tools_path, records, tools):
    """Review is semantic evidence, not a guarantee inferred from non-gold IDs."""
    data = json.loads(Path(path).read_text())
    if data.get('version') != 1 or data.get('scope') != 'TRAIN_record_hard_candidates':
        raise ValueError('Expected reviewed TRAIN candidates')
    for name, actual in [('train', prepared/'records.jsonl'),
                         ('dev', prepared/'records.dev.jsonl'), ('tools', tools_path)]:
        if data['input_sha256'][name] != file_hash(actual):
            raise ValueError('Candidate provenance changed: ' + name)
    if data.get('mix') != {'model': 4, 'lexical': 3, 'random': 8, 'total': 16}:
        raise ValueError('Unsupported candidate mixture')
    mining = Path(path).parent / data['mining_directory']
    complete = json.loads((mining/'COMPLETE.json').read_text())
    for name, field in [('REPORT.json', 'report_sha256'), ('records.jsonl', 'records_sha256')]:
        if file_hash(mining/name) != complete[field]:
            raise ValueError('Mining artifact is unsealed or changed')
    report = json.loads((mining/'REPORT.json').read_text())
    if report['mode'] != 'train-mine' or report['checkpoint_sha256'] != data['mining_checkpoint_sha256']:
        raise ValueError('Candidate source is not the declared TRAIN mining run')
    for name in ('train', 'dev', 'tools'):
        if report['input_sha256'][name] != data['input_sha256'][name]:
            raise ValueError('Mining used different prepared inputs')
    mined = {}
    for line in (mining/'records.jsonl').read_text().splitlines():
        row = json.loads(line); key = stable(row['id'])
        if key in mined: raise ValueError('Duplicate mined record')
        mined[key] = row
    if len(mined) != complete['n'] or len(mined) != report['n']:
        raise ValueError('Incomplete mining coverage')
    index = {record_key(r): r for r in records}
    if len(index) != len(records):
        raise ValueError('Duplicate TRAIN identities')
    overrides, protected, reviewed_keys = {}, {}, set()
    decisions = {'confirmed_inapplicable', 'valid_alternative', 'alias_or_version',
                 'other_unfinished_subtask', 'uncertain', 'pending'}
    for review in data['records']:
        key = stable(review['id'])
        if key not in index or key in reviewed_keys:
            raise ValueError('Non-TRAIN or duplicate candidate record')
        reviewed_keys.add(key)
        record = index[key]
        if record['mode'] != 'tool' or not record['masks']['selection']:
            raise ValueError('Candidate override must be selection-supervised')
        if review['record_sha256'] != fingerprint(record):
            raise ValueError('TRAIN record changed')
        if key not in mined or mined[key]['record_sha256'] != fingerprint(record):
            raise ValueError('No model-scored version of this TRAIN record')
        approved, excluded, candidate_keys = [], [], set()
        ranking = {x['identity']:x['score'] for x in mined[key]['reference_full']['top20']}
        for item in review['candidates']:
            identity = item['identity']
            if identity not in tools or identity in {record['selected'], '__toolbench_finish__'}:
                raise ValueError('Invalid negative identity')
            if identity in candidate_keys:
                raise ValueError('Duplicate reviewed candidate')
            candidate_keys.add(identity)
            if item['decision'] not in decisions:
                raise ValueError('Unknown review decision')
            if item.get('source') != 'model_scored_train' or not item.get('mining_row_sha256'):
                raise ValueError('Missing model-mining provenance')
            if item['mining_row_sha256'] != fingerprint(mined[key]):
                raise ValueError('Model-score row changed')
            if identity not in ranking:
                raise ValueError('Candidate was not in the mined model ranking')
            if not math.isfinite(item['score']) or item['score'] != ranking[identity]:
                raise ValueError('Reviewed score differs from frozen model ranking')
            if item['decision'] != 'confirmed_inapplicable':
                excluded.append(identity)
                continue
            evidence = item.get('review', {})
            for field in ('reason', 'current_step', 'candidate_capability', 'alternative_api_check', 'alias_check'):
                if not isinstance(evidence.get(field), str) or not evidence[field].strip():
                    raise ValueError('Missing semantic review: ' + field)
            approved.append(identity)
        if approved:
            overrides[key] = approved
        if excluded:
            protected[key] = excluded
    if not overrides:
        raise ValueError('No confirmed TRAIN candidates; do not run an unchanged treatment')
    return overrides, {'manifest_sha256': file_hash(path), 'version': 1,
                       'mix': data['mix'], 'records': len(overrides),
                       'protected_records':len(protected),
                       'negative_policy':'exclude all reviewed non-confirmed candidates per record from every sampling channel',
                       'mining_checkpoint_sha256': data['mining_checkpoint_sha256']}, protected


def select_candidates(record, tools, hard, count, seed, overrides=None, protected=None):
    from .toolbench_curriculum_data import candidate_ids
    approved = (overrides or {}).get(record_key(record), [])
    excluded = set((protected or {}).get(record_key(record), []))
    gold = record['selected']
    if not excluded <= set(tools) or gold in excluded or excluded.intersection(approved):
        raise ValueError('Invalid or contradictory per-record candidate protection')
    if not approved:
        original = candidate_ids(record, tools, hard, count, seed)
        if not excluded.intersection(original):
            return original
        eligible = sorted(set(tools)-excluded-set(original)-{'__toolbench_finish__'})
        count_replace = sum(x in excluded for x in original)
        if len(eligible) < count_replace:raise ValueError('Insufficient unprotected replacement candidates')
        replacements = iter(random.Random(f'selection-protection-v1:{seed}:{record_key(record)}').sample(eligible,count_replace))
        return [next(replacements) if x in excluded else x for x in original]
    if count != 16:
        raise ValueError('Reviewed selection training is fixed at 16 candidates')
    if len(set(approved)) != len(approved) or any(x not in tools or x in {gold, '__toolbench_finish__'} for x in approved):
        raise ValueError('Invalid reviewed candidate pool')
    rng = random.Random(f'selection-v1:{seed}:{record_key(record)}')
    model = rng.sample(approved, min(4, len(approved)))
    chosen = {gold, *model}
    lexical = sorted(set(hard.get(gold, [])) - chosen - excluded - {'__toolbench_finish__'})
    if not set(lexical) <= set(tools):
        raise ValueError('Unbound lexical candidate')
    # Keep eight random positions when enough lexical candidates exist.
    chosen.update(rng.sample(lexical, min(7-len(model), len(lexical))))
    rest = sorted(set(tools) - chosen - excluded - {'__toolbench_finish__'})
    if len(rest) < count-len(chosen):
        raise ValueError('Registry too small')
    chosen.update(rng.sample(rest, count-len(chosen)))
    result = sorted(chosen)
    rng.shuffle(result)
    return result
