"""Bound temporary tensors without dropping documents, targets, or gradients."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

def document_groups(lengths, token_budget=4096, max_documents=4):
    groups=[]; group=[]
    for i in sorted(range(len(lengths)), key=lambda i:lengths[i]):
        if group and ((len(group)+1)*lengths[i]>token_budget or len(group)>=max_documents):
            groups.append(group); group=[]
        group.append(i)
    if group: groups.append(group)
    return groups

def target_cross_entropy(head, hidden, target, chunk_size=128):
    hidden=hidden.reshape(-1,hidden.shape[-1]); target=target.reshape(-1)
    def loss_chunk(h,y):
        logits=head(h)
        return F.cross_entropy(logits.float(),y,reduction="none")
    losses=[]
    for start in range(0,len(target),chunk_size):
        h=hidden[start:start+chunk_size];y=target[start:start+chunk_size]
        if len(target)>chunk_size and torch.is_grad_enabled() and (h.requires_grad or any(p.requires_grad for p in head.parameters())):
            value=checkpoint(loss_chunk,h,y,use_reentrant=False)
        else: value=loss_chunk(h,y)
        losses.append(value)
    return torch.cat(losses)
