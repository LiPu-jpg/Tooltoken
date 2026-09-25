"""Verify the downloadable samples against the original ordered training contract."""
import gzip
import hashlib
import json

from rebuild import ROOT, digest_file, load_json, require, verify_contract


def main():
    order, _ = verify_contract()
    report = load_json('PUBLISHED_DATA.json')
    archive = ROOT / report['file']
    require(digest_file(archive) == report['published_archive_sha256'], 'Archive hash mismatch')
    changed = {r['presentation']: r for r in report['affected_presentations']}
    require(len(changed) == 2, 'Unexpected replacement count')
    content_hash = hashlib.sha256()
    count = 0
    with gzip.open(archive, 'rb') as stream:
        for i, line in enumerate(stream):
            require(i < len(order), 'Extra sample')
            content_hash.update(line)
            row = order[i]
            original_hash = row['record_sha256']
            if i in changed:
                require(changed[i]['original_record_sha256'] == original_hash, 'Original sample mismatch')
                expected_hash = changed[i]['published_record_sha256']
                require(line.count(report['replacement'].encode()) == changed[i]['occurrences'], 'Replacement mismatch')
            else:
                expected_hash = original_hash
            require(hashlib.sha256(line.rstrip(b'\n')).hexdigest() == expected_hash, 'Sample hash mismatch: ' + str(i))
            record = json.loads(line)
            for key in ['id', 'query_id', 'source_record_index', 'selected', 'mode', 'masks', 'intent_supervision']:
                require(record.get(key) == row[key], 'Training metadata mismatch: ' + str(i))
            count += 1
    require(count == len(order) == report['record_presentations'], 'Sample count mismatch')
    require(content_hash.hexdigest() == report['published_content_sha256'], 'Content hash mismatch')
    print(json.dumps({'verified': True, 'record_presentations': count,
                      'credential_replacement_presentations': len(changed),
                      'unchanged_presentations': count - len(changed)}))


if __name__ == '__main__':
    main()
