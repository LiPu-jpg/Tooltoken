"""Stream traces, storing the unchanged candidate documents only once.

This is a storage format, not a change to the candidate set or judge input.
Consumers must restore available_tools from the sidecar for each answer.
"""
import copy
import json
from .toolbench_eval_format import convert_trace, evaluator_names


class SharedToolExport:
    def __init__(self, output, tools, bindings):
        self.output, self.tools, self.bindings = output, tools, bindings
        self.names = evaluator_names(tools, bindings)
        self.seen = set()

    def __enter__(self):
        paths = [self.output / n for n in (
            'available_tools.shared.json', 'tooleval_answers.json', 'EXPORT_COMPLETE.json')]
        if any(p.exists() for p in paths):
            raise FileExistsError('Refusing to overwrite export evidence')
        available = [{'name': self.names[i], 'description': t.document,
                      'parameters': copy.deepcopy(t.parameters)}
                     for i, t in sorted(self.tools.items())]
        with paths[0].open('x') as f:
            json.dump(available, f, ensure_ascii=False)
        self.handle = paths[1].open('x')
        self.handle.write('{')
        return self

    def append(self, identity, query, trace):
        if not isinstance(identity, str) or identity in self.seen:
            raise ValueError('Invalid or duplicate query ID')
        answer = convert_trace(query, trace, self.tools, self.bindings,
                               names=self.names, include_available=False)
        if self.seen:
            self.handle.write(',\n')
        self.handle.write(json.dumps(identity) + ':')
        json.dump(answer, self.handle, ensure_ascii=False)
        self.handle.flush()
        self.seen.add(identity)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.handle.write('}\n')
        finally:
            self.handle.close()
        if exc_type is None:
            with (self.output / 'EXPORT_COMPLETE.json').open('x') as f:
                json.dump({'format': 'shared-tools-v1', 'episodes': len(self.seen),
                           'available_tools': 'available_tools.shared.json',
                           'restore_sidecar_before_upstream_evaluation': True}, f)
