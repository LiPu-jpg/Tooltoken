"""Five latent candidates for selection, selected documentation for exact inputs."""
import hashlib
import torch
from .toolbench_topk import TopKCurriculumAgent
from .toolbench_agent import GenerationFailure

INTERFACE = 'native-toolbench-top5-selected-document-v41'
RECIPE = {'version': 1, 'selection': 'unchanged_top5_memory_reader',
          'arguments': 'selected_full_registration_document',
          'repair': 'selected_full_registration_document',
          'information': 'unchanged_memory_readback',
          'new_parameters': False}


class HybridDocumentAgent(TopKCurriculumAgent):
    interface_version = INTERFACE

    def prefix(self, history, task, thought='', *, memory=None, document='',
               condition=None, **kwargs):
        if task in {'arguments', 'repair'}:
            if not document:
                raise ValueError('Hybrid argument/repair path requires bound documentation')
            condition = 'full_document'
        return super().prefix(history, task, thought, memory=memory,
                              document=document, condition=condition, **kwargs)

    def select_from_ranking(self, history, intent, ranking):
        identity, details = super().select_from_ranking(history, intent, ranking)
        self._last_reader_trace = dict(details)
        return identity, details

    def reader_scores(self, history, intent, memories):
        scores = super().reader_scores(history, intent, memories)
        if not torch.is_grad_enabled():
            self._last_reader_logits = scores.detach().float().cpu().reshape(-1).tolist()
        return scores

    def decide(self, history, tools, **kwargs):
        self._last_reader_trace = None
        self._last_reader_logits = None
        try:
            decision = super().decide(history, tools, **kwargs)
        except GenerationFailure as exc:
            if self._last_reader_trace is not None:
                exc.details['reader_trace'] = {
                    **self._last_reader_trace, 'logits': self._last_reader_logits}
            identity = exc.details.get('api_identity')
            if identity in tools:
                exc.details['selected_document_sha256'] = hashlib.sha256(
                    tools[identity].registration_document.encode()).hexdigest()
            raise
        decision.generation_budgets.update(interface=INTERFACE)
        if self._last_reader_trace is not None:
            decision.generation_budgets['reader_logits'] = self._last_reader_logits
        if decision.api_identity in tools:
            decision.generation_budgets['selected_document_sha256'] = hashlib.sha256(
                tools[decision.api_identity].registration_document.encode()).hexdigest()
        return decision
