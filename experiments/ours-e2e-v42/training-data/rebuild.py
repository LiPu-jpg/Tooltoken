"""Authenticate the Ours data contract and reconstruct checkpoint-contributing examples.

No model, GPU, network, or third-party Python package is needed. Raw trajectories
can contain benchmark credential literals; reconstructed output is local data.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def open_bytes(path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')


def digest_file(path, decompress=False):
    digest = hashlib.sha256()
    with (open_bytes(path) if decompress else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def load_json(name):
    return json.loads((ROOT / name).read_text())


def verify_contract():
    manifest = load_json('MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), 'Invalid manifest path')
        require(path.stat().st_size == expected['bytes'], 'Wrong file size: ' + name)
        require(digest_file(path) == expected['sha256'], 'Wrong file hash: ' + name)
        if 'uncompressed_sha256' in expected:
            require(digest_file(path, True) == expected['uncompressed_sha256'], 'Wrong content hash: ' + name)
    with gzip.open(ROOT / 'sample-order.jsonl.gz', 'rt') as stream:
        order = [json.loads(line) for line in stream]
    require(len(order) == 43200, 'Incomplete order')
    require([r['presentation'] for r in order] == list(range(43200)), 'Changed presentation order')
    require([r['update'] for r in order] == [i // 8 + 1 for i in range(43200)], 'Changed update membership')
    lineage = load_json('lineage.json')
    require(lineage['checkpoint_manifest_sha256'] == manifest['checkpoint_manifest_sha256'], 'Checkpoint mismatch')
    segments = {s['id']: s for s in lineage['segments']}
    for row in order:
        segment = segments[row['segment']]
        require(segment['start_update'] < row['update'] <= segment['end_update'], 'Wrong segment')
        require(0 <= row['rank'] < segment['world_size'], 'Wrong rank')
    registry = set(load_json('training-registry-api-ids.json'))
    positives = {r['selected'] for r in order if r['mode'] == 'tool' and r['masks'].get('selection', 0) > 0}
    selected = {r['selected'] for r in order}
    require(positives == set(load_json('positive-seen-api-ids.json')), 'Positive API membership mismatch')
    require(selected == set(load_json('selected-document-api-ids.json')), 'Document association mismatch')
    require(selected <= registry and len(registry) == 48318, 'Training registry mismatch')
    summary = load_json('SUMMARY.json')
    require(len(positives) == summary['ordinary_tool_identities'] == 7171, 'Positive API count mismatch')
    require(len({tuple(r['id']) for r in order}) == summary['unique_prepared_decision_ids'] == 42832, 'Decision count mismatch')
    require(len({r['query_id'] for r in order}) == summary['unique_query_ids'] == 31289, 'Query count mismatch')
    require(dict(collections.Counter(r['mode'] for r in order)) == summary['mode_counts'], 'Mode counts mismatch')
    with gzip.open(ROOT / 'train-tools.jsonl.gz', 'rt') as stream:
        tool_ids = [json.loads(line)['api_identity'] for line in stream]
    require(len(tool_ids) == len(set(tool_ids)) and set(tool_ids) == registry, 'Tool document membership mismatch')
    inputs = load_json('SOURCE_INPUTS.json')
    require(digest_file(ROOT / 'train-tools.jsonl.gz', True) == load_json('prepared-base.READY.json')['train_tools.jsonl'], 'Original tool hash mismatch')
    require(digest_file(ROOT / 'hard-negatives.json.gz', True) == load_json('prepared-base.READY.json')['hard-negatives.json'], 'Original negative hash mismatch')
    return order, inputs


def read_records(path, expected_hash, wanted, wrapped=False):
    digest = hashlib.sha256()
    records = {}
    count = 0
    with open_bytes(path) as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            value = json.loads(raw)
            index = value['record_index'] if wrapped else count
            if index in wanted:
                require(index not in records, 'Duplicate input index')
                records[index] = value['record'] if wrapped else value
            count += 1
    require(digest.hexdigest() == expected_hash, 'Source input hash mismatch: ' + str(path))
    require(set(records) == wanted, 'Missing required source records: ' + str(path))
    return records, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--base-records', type=Path, help='Original full 173746-record corpus, plain JSONL or gzip')
    group.add_argument('--base-selection', type=Path, help='Authenticated records-base.jsonl.gz collected from HPC')
    parser.add_argument('--task-state-records', type=Path, help='Original 1600-record v42 corpus, plain JSONL or gzip')
    parser.add_argument('--through-update', type=int, choices=[5000, 5200, 5400], default=5400)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    order, inputs = verify_contract()
    if args.verify_only:
        print(json.dumps({'verified': True, 'presentations': len(order), 'raw_examples_verified': False}))
        return
    if not (args.base_records or args.base_selection) or not args.output_dir:
        parser.error('Provide --base-records or --base-selection, and a fresh --output-dir')
    if args.through_update == 5400 and not args.task_state_records:
        parser.error('--task-state-records is required for the 5400-update checkpoint')
    require(not args.output_dir.exists(), 'Output directory already exists')
    chosen = [r for r in order if r['update'] <= args.through_update]
    wanted = {r['index'] for r in chosen if r['pool'] == 'base'}
    key = 'base_selection_archive' if args.base_selection else 'base'
    info = inputs[key]
    base, count = read_records(args.base_selection or args.base_records,
                               info.get('uncompressed_sha256', info.get('sha256')),
                               wanted, wrapped=bool(args.base_selection))
    require(count == info['records'], 'Wrong base corpus count')
    pools = {'base': base}
    if args.through_update == 5400:
        state, count = read_records(args.task_state_records, inputs['task_state']['sha256'],
                                    {r['index'] for r in chosen if r['pool'] == 'task-state'})
        require(count == 1600, 'Wrong task-state corpus count')
        pools['task-state'] = state
    for row in chosen:
        record = pools[row['pool']][row['index']]
        require(hashlib.sha256(canonical(record)).hexdigest() == row['record_sha256'], 'Record fingerprint mismatch')
        require(record['split'] == 'train' and record['synthetic'] is False, 'Unexpected split or record origin')
        for key in ['id', 'query_id', 'source_record_index', 'selected', 'mode', 'masks', 'intent_supervision']:
            require(record.get(key) == row[key], 'Record metadata mismatch: ' + key)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / 'records.jsonl.gz'
    content_hash = hashlib.sha256()
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in chosen:
                line = canonical(pools[row['pool']][row['index']]) + b'\n'
                content_hash.update(line)
                stream.write(line)
    report = {'verified': True, 'through_update': args.through_update, 'record_presentations': len(chosen),
              'positive_apis': len({r['selected'] for r in chosen if r['mode'] == 'tool' and r['masks']['selection'] > 0}),
              'records_content_sha256': content_hash.hexdigest(), 'archive_sha256': digest_file(output),
              'contract_manifest_sha256': digest_file(ROOT / 'MANIFEST.json'),
              'order': 'update then recorded micro/rank; retain repeated IDs and altered supervision',
              'PORTS_training_alignment_verified': False}
    (args.output_dir / 'RECONSTRUCTION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
