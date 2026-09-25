"""Verify export hashes, then run CPU suites in isolated Python processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for relative, entry in manifest['files'].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != entry.get('export_sha256', entry['sha256']):
            raise ValueError('Export source changed: ' + relative)
    print(f"Verified {len(manifest['files'])} source files", flush=True)
    source = str(ROOT / 'release-v1/source/src')
    environment = {**os.environ, 'PYTHONPATH': source, 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    for suite, extra in [
        ('release-v1/tests', None),
        ('eval-reader-ab-v1/scripts/test_reader.py', 'eval-reader-ab-v1/variant'),
        ('repairs/completion-v6-source-patch-20260923/tests', None),
    ]:
        env = dict(environment)
        if extra:
            env['PYTHONPATH'] += os.pathsep + str(ROOT / extra)
        subprocess.run([sys.executable, '-m', 'pytest', '-q', suite], cwd=ROOT, env=env, check=True)
    for protocol in ('native', 'reader', 'completion-v6'):
        subprocess.run([sys.executable, str(ROOT / 'run.py'), '--protocol', protocol, '--help'],
                       cwd=ROOT, env=environment, check=True, stdout=subprocess.DEVNULL)
    print('All source, CPU regression and CLI checks passed.', flush=True)


if __name__ == '__main__':
    main()
