"""Portable entrypoint for the exported ours E2E protocols."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--protocol', choices=['native', 'reader', 'completion-v6'], default='native')
    parser.add_argument('--reader-mode', choices=['memory_fp32', 'documents_fp32'])
    options, remaining = parser.parse_known_args()
    helping = '-h' in remaining or '--help' in remaining
    if not helping and options.protocol != 'native' and options.reader_mode is None:
        parser.error('--reader-mode is required for reader and completion-v6')
    if options.protocol == 'native' and options.reader_mode:
        parser.error('--reader-mode requires --protocol reader or completion-v6')
    sys.path.insert(0, str(ROOT / 'release-v1/source/src'))
    sys.argv = [sys.argv[0], *remaining]
    if options.protocol == 'completion-v6':
        if options.reader_mode:
            sys.argv += ['--reader-mode', options.reader_mode]
        runpy.run_path(str(ROOT / 'repairs/completion-v6-source-patch-20260923/scripts/run_native_variant.py'), run_name='__main__')
        return

    from latent_register import run_toolbench_e2e as runner
    if options.protocol == 'reader' and not helping:
        variant = ROOT / 'eval-reader-ab-v1/variant'
        sys.path.insert(0, str(variant))
        from reader_variant import install
        original_load = runner.load_agent
        evidence = {}

        def load_with_reader(*args, **kwargs):
            agent = original_load(*args, **kwargs)
            tools_path = Path(remaining[remaining.index('--tools') + 1])
            split = remaining[remaining.index('--split') + 1]
            tools = runner.load_tools(tools_path, split=split)
            install(agent, tools, options.reader_mode)
            evidence.update(mode=options.reader_mode, training=False,
                            checkpoint_context_limit=agent.limits.context,
                            overflow='explicit error; no truncation or fallback',
                            projection='FP32 with autocast disabled',
                            adapter_sha256=hashlib.sha256((variant / 'reader_variant.py').read_bytes()).hexdigest(),
                            tools_sha256=runner.sha256(tools_path))
            return agent

        runner.load_agent = load_with_reader
        try:
            runner.main()
            output = Path(remaining[remaining.index('--output-dir') + 1])
            provenance = json.loads((output / 'PROVENANCE.json').read_text())
            provenance['reader_variant'] = evidence
            with (output / 'READER_PROVENANCE.json').open('x') as handle:
                json.dump(provenance, handle, indent=2)
        finally:
            runner.load_agent = original_load
    else:
        runner.main()


if __name__ == '__main__':
    main()
