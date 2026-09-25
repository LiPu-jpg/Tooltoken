"""One explicit candidate-recipe migration at a verified optimizer boundary."""
import copy
from .toolbench_continuation import RuntimeTrainingState
from .selection_candidates import fingerprint


class SelectionTrainingState(RuntimeTrainingState):
    def __init__(self, model, contract, transition=None):
        super().__init__(model, contract)
        self.transition = transition
        self.transition_applied = False

    def load_state_dict(self, state):
        saved, current = state['contract'], self.contract
        if saved.get('selection_candidates') == current.get('selection_candidates'):
            return super().load_state_dict(state)
        t = self.transition
        if not t or t.get('version') != 1:
            raise ValueError('Explicit selection transition required')
        if saved.get('selection_candidates') is not None or current.get('selection_candidates') is None:
            raise ValueError('Only original -> reviewed-candidate migration is supported')
        if state['cursor']['updates'] != t['at_update'] or fingerprint(saved) != t['parent_contract_sha256']:
            raise ValueError('Wrong parent contract or optimizer boundary')
        if t['to_candidates_sha256'] != current['selection_candidates']['manifest_sha256']:
            raise ValueError('Wrong candidate version')
        expected = dict(current)
        expected.pop('selection_candidates')
        if expected != saved:
            raise ValueError('Selection migration cannot also change data/order/loss/world size or budget')
        migrated = copy.copy(state)
        migrated['contract'] = dict(current)
        super().load_state_dict(migrated)
        self.transition_applied = True
