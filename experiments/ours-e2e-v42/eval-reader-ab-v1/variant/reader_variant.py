"""Explicit per-agent reader variant. Original checkpoint/source is unchanged."""
from types import MethodType
import torch
from torch.nn import functional as F


def project(hidden, weight, bias=None):
    # Disabling autocast is necessary: casting inputs alone is not sufficient.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())


def score_prefix(agent, prefix, k):
    tokens=[agent.tokenizer.encode(str(i+1),add_special_tokens=False) for i in range(k)]
    if any(len(t)!=1 for t in tokens) or len({t[0] for t in tokens})!=k:
        raise ValueError('Distinct single-token candidate digits required')
    h=agent._hidden(inputs_embeds=prefix,use_cache=False).last_hidden_state[:,-1]
    head=agent.backbone.get_output_embeddings();ix=agent._ids([t[0] for t in tokens])
    w=head.weight.index_select(0,ix)
    b=head.bias.index_select(0,ix) if getattr(head,'bias',None) is not None else None
    scores=project(h,w,b)
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite reader scores')
    if not torch.is_grad_enabled():agent._last_reader_logits=scores.detach().cpu().reshape(-1).tolist()
    return scores


def install(agent, tools, mode):
    """Use after checkpoint loading; mode must be recorded in runtime provenance.

    memory_fp32 updates both reader loss and serving projection. documents_fp32
    is an inference-only input ablation, not a trained full-document baseline.
    """
    if mode not in {'memory_fp32','documents_fp32'}:raise ValueError(mode)
    if hasattr(agent,'reader_variant'):raise ValueError('Reader variant already installed')
    def memory_scores(self,history,intent,memories):
        return score_prefix(self,self.reader_prefix(history,intent,memories),len(memories))
    agent.reader_scores=MethodType(memory_scores,agent)
    if mode=='documents_fp32':
        from latent_register.toolbench_topk import candidate_order
        from latent_register.toolbench_curriculum_data import clean_history
        from latent_register.toolbench_data import compact
        from latent_register.toolbench_agent import SYSTEM
        def select(self,history,intent,ranking):
            if self.training:raise ValueError('Document inference ablation is not a training recipe')
            if len(ranking)<5:raise ValueError('Five candidates required')
            order=candidate_order(clean_history(history,self.history_format),list(ranking[:5]))
            visible=self._with_state(history) if hasattr(self,'_with_state') else history
            body='Causal history JSON:\n'+compact(clean_history(visible,self.history_format))
            body+='\nCurrent causal subtask (not evidence for parameter values):\n'+intent
            for i,key in enumerate(order):body+=f'\nCandidate SLOT={i+1}:\n'+tools[key].registration_document
            body+='\nTask: Read all candidate tool documents and select the tool for the next action. Return only its candidate number (1 to 5), without arguments or explanation.'
            rendered=self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=self.tokenizer.encode(rendered,add_special_tokens=False)
            if len(ids)+1>self.limits.context:raise ValueError('Complete five-document reader exceeds context; no truncation')
            prefix=self.backbone.get_input_embeddings()(self._ids(ids)).unsqueeze(0)
            scores=score_prefix(self,prefix,5);slot=int(scores.argmax(-1).item())
            budget={'reader_candidates':order,'reader_selected_slot':slot+1,'reader_k':5,'reader_memory_positions':0,'reader_document_positions':len(ids),'selection_interface':'top5-documents-fp32-inference-v1','retrieval_top1':ranking[0],'reader_document_hashes':{key:tools[key].document_hash for key in order}}
            self._last_reader_trace=dict(budget)
            return order[slot],budget
        agent.select_from_ranking=MethodType(select,agent)
    agent.reader_variant=mode
    return agent
