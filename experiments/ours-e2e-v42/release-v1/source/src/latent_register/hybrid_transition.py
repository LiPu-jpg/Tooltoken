"""Allow only a declared input-interface migration, retaining all optimizer state."""
import copy
from .topk_transition import TopKTrainingState
from .toolbench_topk import INTERFACE as OLD_INTERFACE, RECIPE as TOPK_RECIPE
from .toolbench_hybrid import INTERFACE, RECIPE
from .selection_candidates import fingerprint


class HybridTrainingState(TopKTrainingState):
    def load_state_dict(self, state):
        saved, current = state['contract'], self.contract
        if saved.get('hybrid_document_reader') == current.get('hybrid_document_reader'):
            return super().load_state_dict(state)
        t = self.transition
        if not t or t.get('kind') != 'selected-document-v1':
            raise ValueError('Explicit selected-document migration required')
        if (state['cursor'] != t['parent_cursor'] or
                state['cursor']['updates'] != t['at_update'] or
                fingerprint(saved) != t['parent_contract_sha256']):
            raise ValueError('Wrong hybrid parent/cursor')
        if (saved.get('interface') != OLD_INTERFACE or
                saved.get('topk_reader') != TOPK_RECIPE or
                saved.get('hybrid_document_reader') is not None or
                current.get('interface') != INTERFACE or
                current.get('hybrid_document_reader') != RECIPE):
            raise ValueError('Unsupported hybrid interface change')
        expected = dict(current)
        expected.pop('hybrid_document_reader')
        expected['interface'] = OLD_INTERFACE
        if expected != saved:
            raise ValueError('Hybrid migration cannot change data/loss/world size/schedule')
        migrated = copy.copy(state)
        migrated['contract'] = dict(current)
        super().load_state_dict(migrated)
        self.transition_applied = True
