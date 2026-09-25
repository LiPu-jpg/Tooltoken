"""Deterministic per-tool schema coverage, independent of rank and batching."""
from collections import Counter, defaultdict
import hashlib
from .toolbench_curriculum_data import information_targets
from .toolbench_data import compact


def targets(tool):
    old = information_targets(tool)
    signatures = [t for t in old if 'signature' in t]
    fields = [t for t in old if 'path' in t]
    capability = [t for t in old if 'capability' in t]
    return [{'required_fields': tool.parameters.get('required', [])}] + signatures + fields + capability


def request(fact):
    if 'required_fields' in fact:return 'all required top-level field names, exactly as declared'
    if 'signature' in fact:return 'complete input signature'
    if 'path' in fact:return 'field at path ' + compact(fact['path'])
    if 'capability' in fact:return 'capability'
    raise ValueError('Unknown schema fact')


def replay_plan(records, tools, count, cursor=0):
    """Breadth-first facts over unique TRAIN tools; cursor resumes across rounds.

    The caller must persist the cursor together with the pool hash. Dataset row
    order and duplicated queries must not bias which tool gets schema replay.
    """
    if count < 0 or cursor < 0:
        raise ValueError('Nonnegative schema count/cursor required')
    representatives = {}
    for record in sorted(records, key=lambda r: str(r['id'])):
        if record['selected'] in tools:
            representatives.setdefault(record['selected'], record)
    identities = sorted(representatives, key=lambda identity: hashlib.sha256(
        ('native-v30-schema-replay\0' + identity).encode()).digest())
    facts = {identity: targets(tools[identity]) for identity in identities}
    pool = [(identity, index) for index in range(max(map(len, facts.values()), default=0))
            for identity in identities if index < len(facts[identity])]
    if not pool:
        raise ValueError('Empty TRAIN schema replay pool')
    entries = []
    for position in range(cursor, cursor + count):
        identity, index = pool[position % len(pool)]
        entries.append((representatives[identity], index, facts[identity][index], position))
    contract = {'cursor_start': cursor, 'cursor_next': cursor + count,
                'pool_size': len(pool), 'unique_tools': len(identities),
                'pool_sha256': hashlib.sha256(compact([
                    [identity, index, facts[identity][index]] for identity, index in pool]).encode()).hexdigest()}
    return entries, contract


def annotate_schedule(schedule, records, tools):
    counts = Counter(); coverage = defaultdict(Counter)
    result = []
    for entry in schedule:
        identity = records[entry['record_index']]['selected']
        facts = targets(tools[identity]); index = counts[identity] % len(facts)
        counts[identity] += 1; coverage[identity][index] += 1
        result.append({**entry, 'information_index': index})
    report = {identity: {'planned_exposures': counts[identity], 'target_count': len(targets(tools[identity])),
                       'planned_target_exposures': dict(hist), 'planned_all_targets_covered': len(hist)==len(targets(tools[identity]))}
              for identity,hist in coverage.items()}
    return result, report
