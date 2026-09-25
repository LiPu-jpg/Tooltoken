"""Audited single interface/loss migration; optimizer tensors remain untouched."""
import copy
from .selection_transition import SelectionTrainingState
from .selection_candidates import fingerprint
from .toolbench_topk import INTERFACE, RECIPE
from .toolbench_curriculum import INTERFACE as OLD_INTERFACE

class TopKTrainingState(SelectionTrainingState):
    def load_state_dict(self,state):
        if state['contract'].get('topk_reader')==self.contract.get('topk_reader'):
            return super().load_state_dict(state)
        t=self.transition
        if not t or t.get('kind')!='topk-reader-v1':raise ValueError('Explicit TopK transition required')
        saved=state['contract'];current=self.contract
        if state['cursor']['updates']!=t['at_update'] or fingerprint(saved)!=t['parent_contract_sha256']:
            raise ValueError('Incorrect TopK parent contract/update')
        if state['cursor']!=t['parent_cursor']:raise ValueError('Incorrect parent loader cursor')
        if saved.get('topk_reader') is not None or saved.get('interface')!=OLD_INTERFACE:
            raise ValueError('Only original reader -> TopK reader migration allowed')
        if current.get('topk_reader')!=RECIPE or current.get('interface')!=INTERFACE:
            raise ValueError('Unrecognized TopK recipe')
        expected=dict(current);expected.pop('topk_reader');expected['interface']=OLD_INTERFACE
        if expected!=saved:raise ValueError('TopK migration may only change interface and reader recipe')
        migrated=copy.copy(state);migrated['contract']=dict(current)
        super().load_state_dict(migrated)
        self.transition_applied=True
