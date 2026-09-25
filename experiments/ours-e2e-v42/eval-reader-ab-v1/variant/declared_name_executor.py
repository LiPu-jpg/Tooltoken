"""Opt-in, model-independent transport variant; never fills or fixes values."""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from name_contract import translate


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=4)
def load_manifest(path, expected_hash):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError('Mapping manifest hash mismatch')
    value = json.loads(data)
    if value['policy'] != 'declared-parameter-names-v1':
        raise ValueError('Wrong mapping policy')
    return value


class DeclaredNameExecutor:
    def __init__(self, base, manifest, bindings, manifest_hash):
        self.base, self.manifest, self.manifest_hash = base, manifest, manifest_hash
        self.identities = set(bindings) - {'__toolbench_finish__'}
        known = set(manifest['entries']) | set(manifest['quarantine'])
        if not self.identities <= known:
            raise ValueError('Executor identity missing from mapping registry')
        for identity in self.identities:
            entry = manifest['entries'].get(identity, manifest['quarantine'].get(identity))
            if entry['source_binding'] != bindings[identity]:
                raise ValueError('Executor source binding differs from mapping source')

    def __call__(self, identity, arguments):
        if identity not in self.identities or not isinstance(arguments, dict):
            raise ValueError('Require bound identity and argument object')
        entry = self.manifest['entries'].get(identity)
        if entry is None:
            # Preserve historical behavior for ambiguous metadata; never guess a contract.
            wire = json.loads(json.dumps(arguments, ensure_ascii=False, allow_nan=False))
            status = 'quarantined_unchanged'
        else:
            mapping = entry['mapping']
            if set(arguments) - set(mapping):
                # Caller may emit a declared name after actual tool feedback, or
                # an extra property allowed by an open schema. Do not invent an
                # alias, drop it, or add a new pre-execution rejection. Preserve
                # the entire request and let the original executor respond.
                wire = json.loads(json.dumps(arguments, ensure_ascii=False, allow_nan=False))
                status = 'unmapped_keys_unchanged'
            else:
                wire = translate(arguments, mapping)
                status = 'renamed' if wire != arguments else 'certified_unchanged' 
        result = self.base(identity, wire)
        extra = {'parameter_name_policy': 'declared-parameter-names-v2',
                 'parameter_mapping_registry_policy': self.manifest['policy'], 'mapping_manifest_sha256': self.manifest_hash,
                 'parameter_mapping_status': status, 'canonical_arguments_sha256': digest(arguments),
                 'wire_arguments_sha256': digest(wire), 'fills_missing_fields': False}
        return replace(result, metadata={**(result.metadata or {}), **extra})


def create_executor(*, query_id, config, tool_bindings):
    from latent_register.toolbench_http import create_executor as original_factory
    base = original_factory(query_id=query_id, config=config, tool_bindings=tool_bindings)
    if not config.get('enable_declared_parameter_names', False):
        return base
    if config['backend_kind'] not in {'stabletoolbench_virtual', 'loopback_contract_only'}:
        raise ValueError('Variant not certified for real API services')
    manifest = load_manifest(config['parameter_mapping_path'], config['parameter_mapping_sha256'])
    if manifest['tools_sha256'] != config['parameter_mapping_tools_sha256'] or \
       manifest['documents_sha256'] != config['parameter_mapping_documents_sha256']:
        raise ValueError('Registry/document version mismatch')
    return DeclaredNameExecutor(base, manifest, tool_bindings, config['parameter_mapping_sha256'])
