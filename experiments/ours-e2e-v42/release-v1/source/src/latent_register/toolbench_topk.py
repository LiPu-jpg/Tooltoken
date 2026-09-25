"""Top-5 latent-memory reader; no new trainable parameters or vocabulary rows."""
import hashlib
import torch
from torch.nn import functional as F
from .toolbench_lora import OptimizedCurriculumAgent
from .toolbench_agent import SYSTEM, MEMORY_MARKER
from .toolbench_curriculum_data import clean_history
from .toolbench_data import compact

INTERFACE = 'native-toolbench-top5-memory-reader-v40'
RECIPE = {'version':1, 'k':5, 'reader_loss_weight':1.0,
          'train_candidates':'sampled16_top5_gold_replace_lowest_when_missing',
          'inference_candidates':'full_registry_top5_no_gold',
          'order':'sha256_causal_history_and_identity',
          'output':'ordinary_digit_1_to_5_constrained_logits',
          'arguments':'selected_single_memory_after_reader_choice'}


def candidate_order(history, identities):
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('Distinct candidate identities required')
    context = compact(history)
    return sorted(identities, key=lambda x:(hashlib.sha256((context+'\0'+x).encode()).hexdigest(),x))


def shortlist(scores, identities, history, *, gold=None, k=5):
    if len(identities) < k or len(set(identities)) != len(identities):
        raise ValueError('Insufficient/distinct retrieval candidates')
    if scores.shape != (1,len(identities)) or not torch.isfinite(scores).all():
        raise ValueError('Invalid retrieval scores')
    ranked = sorted(range(len(identities)),key=lambda i:(-float(scores[0,i].detach()),identities[i]))
    chosen = [identities[i] for i in ranked[:k]]
    gold_present = gold in chosen if gold is not None else None
    if gold is not None:
        if gold not in identities: raise ValueError('Unbound training gold')
        if not gold_present: chosen[-1] = gold
    return candidate_order(history,chosen), gold_present


def reader_parts(tokenizer, history, intent, k, history_format='legacy'):
    if not 1 <= k <= 5: raise ValueError('Reader supports at most five slots')
    body='Causal history JSON:\n'+compact(clean_history(history,history_format))
    body+='\nCurrent causal subtask (not evidence for parameter values):\n'+intent
    if MEMORY_MARKER in body: raise ValueError('Reserved marker in source')
    for i in range(k): body+=f'\nCandidate SLOT={i+1}:\n'+MEMORY_MARKER
    body+='\nTask: Read all candidate tool memories and select the tool for the next action. Return only its candidate number (1 to '+str(k)+'), without arguments or explanation.'
    rendered=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':body}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pieces=rendered.split(MEMORY_MARKER)
    if len(pieces)!=k+1:raise ValueError('Reader memory marker count changed')
    return [tokenizer.encode(p,add_special_tokens=False) for p in pieces]


class TopKCurriculumAgent(OptimizedCurriculumAgent):
    interface_version=INTERFACE
    topk_reader=True

    def reader_prefix(self, history, intent, memories):
        k=memories.shape[0]
        embedding=self.backbone.get_input_embeddings()
        if memories.shape != (k,self.slots,embedding.weight.shape[1]):
            raise ValueError('All candidate memory slots must be injected')
        parts=reader_parts(self.tokenizer,history,intent,k,self.history_format)
        chunks=[]
        for i,part in enumerate(parts):
            chunks.append(embedding(self._ids(part)))
            if i<k:chunks.append(memories[i].to(chunks[0].dtype))
        prefix=torch.cat(chunks).unsqueeze(0)
        if prefix.shape[1]+1>self.limits.context:raise ValueError('Complete TopK reader prefix exceeds context; no truncation')
        return prefix

    def reader_scores(self, history, intent, memories):
        tokens=[self.tokenizer.encode(str(i+1),add_special_tokens=False) for i in range(memories.shape[0])]
        if any(len(t)!=1 for t in tokens) or len({t[0] for t in tokens})!=len(tokens):
            raise ValueError('Candidate numbers must be distinct single ordinary tokens')
        prefix=self.reader_prefix(history,intent,memories)
        hidden=self._hidden(inputs_embeds=prefix,use_cache=False).last_hidden_state[:,-1]
        head=self.backbone.get_output_embeddings();index=self._ids([t[0] for t in tokens])
        weight=head.weight.index_select(0,index)
        bias=head.bias.index_select(0,index) if getattr(head,'bias',None) is not None else None
        return F.linear(hidden.to(weight.dtype),weight,bias).float()

    def training_reader_loss(self, record, intent, ids, scores, memories):
        chosen,_=shortlist(scores,ids,clean_history(record['history'],self.history_format),gold=record['selected'])
        picked=torch.stack([memories[ids.index(key)] for key in chosen])
        logits=self.reader_scores(record['history'],intent,picked)
        return F.cross_entropy(logits,self._ids([chosen.index(record['selected'])]))

    def select_from_ranking(self, history, intent, ranking):
        if len(ranking)<5: raise ValueError('TopK serving registry must contain at least five tools')
        chosen=candidate_order(clean_history(history,self.history_format),list(ranking[:5]))
        memory=torch.stack([self._registry[key][2] for key in chosen])
        scores=self.reader_scores(history,intent,memory)
        if not torch.isfinite(scores).all():raise ValueError('Nonfinite reader scores')
        slot=int(scores.argmax(-1).item())
        return chosen[slot],{'reader_candidates':chosen,'reader_selected_slot':slot+1,
            'reader_k':5,'reader_memory_positions':5*self.slots,
            'selection_interface':INTERFACE,'retrieval_top1':ranking[0]}
