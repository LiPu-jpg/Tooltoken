"""Shared Qwen LoRA and both compilers, with an unchanged causal interface.

Only tokenization may be cached across updates. Document hidden states remain
in the current adapter's autograd graph. No optimizer/parameter CPU offload.
"""
import torch
from torch.nn import functional as F
from peft import LoraConfig, get_peft_model
from .toolbench_curriculum import CurriculumAgent
from .toolbench_curriculum_data import CONTROL, candidate_ids, information_targets
from .toolbench_data import compact
from .selection_candidates import select_candidates


def enable_shared_lora(backbone, rank=16, alpha=32):
    backbone.requires_grad_(False)
    if hasattr(backbone, 'peft_config'):
        raise ValueError('Refuse nested or doubled LoRA')
    return get_peft_model(backbone, LoraConfig(r=rank, lora_alpha=alpha,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0., bias='none', task_type='CAUSAL_LM'))


def train_lora_and_compilers(agent):
    if not hasattr(agent.backbone, 'peft_config'):
        raise ValueError('Expected shared adapter checkpoint')
    agent.requires_grad_(False)
    for name, param in agent.backbone.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            param.requires_grad_(True)
    agent.output_compiler.requires_grad_(True)
    agent.memory_compiler.requires_grad_(agent.condition == 'memory')
    agent.train()


def optimizer_groups(agent, lora_lr, compiler_lr):
    groups = [{'params': [p for p in agent.backbone.parameters() if p.requires_grad], 'lr': lora_lr},
              {'params': [p for n,p in agent.named_parameters() if not n.startswith('backbone.') and p.requires_grad],
               'lr': compiler_lr}]
    if not all(g['params'] for g in groups):
        raise ValueError('Missing adapter or compiler trainable parameters')
    if any(p.requires_grad for n,p in agent.backbone.named_parameters()
           if '.lora_A.' not in n and '.lora_B.' not in n):
        raise ValueError('Static Qwen parameters must remain frozen')
    return groups


class OptimizedCurriculumAgent(CurriculumAgent):
    """Skip zero-weight tasks; resample only the selected tool's memory.

    DDP uses find_unused_parameters=True. Unlike ZeRO parameter gathering,
    it does not require identical forward module traversal on every rank.
    """
    def compile(self, tools, *, profile=False, memory_indices=None):
        if not self.training or len(tools)<=4 or memory_indices is None:
            return super().compile(tools, profile=profile, memory_indices=memory_indices)
        # Length grouping changes padding, never the candidates, order of scores,
        # complete document contents, or number of optimizer updates.
        if profile: raise ValueError('Registration profiling is an eval-only path')
        lengths=[len(self.tokenizer.encode('Represent this tool for registration.\n'+t.registration_document,
                                         add_special_tokens=False)) for t in tools]
        order=sorted(range(len(tools)), key=lambda i:lengths[i])
        row_map={}; memory_map={}
        from .training_memory import document_groups
        for group in document_groups(lengths):
            selected=[i for i in group if i in memory_indices]
            rows,memories=super().compile([tools[i] for i in group],
                memory_indices=[group.index(i) for i in selected])
            row_map.update({i:rows[j] for j,i in enumerate(group)})
            memory_map.update({i:memories[j] for j,i in enumerate(selected)})
        result=torch.stack([row_map[i] for i in range(len(tools))])
        memory=(torch.stack([memory_map[i] for i in memory_indices]) if memory_indices
                else result.new_empty((0,self.slots,result.shape[-1])))
        return result,memory

    def forward(self, records, tools, *, hard=None, stage=1, candidate_count=16, seed=17, information_index=None, **unused):
        if stage not in (1, 2) or not records:
            raise ValueError('Invalid curriculum stage/records')
        if len(records)==1 and 'supervised_unit' in records[0]:
            from .toolbench_rl import PolicyLearner
            event=records[0]['supervised_unit']
            if not event.get('supervised'):raise ValueError('Supplemental SFT requires supervised units')
            result=PolicyLearner(self)(event,tools)
            losses={k:result['loss'].new_zeros(()) for k in ('selection','arguments','information','control','repair','answer','intent')}
            losses[event['kind']]=result['loss'];losses['loss']=result['loss']
            return losses
        losses = {k: [] for k in ('selection','arguments','information','control','repair','answer','intent')}
        if getattr(self, 'topk_reader', False): losses['reader'] = []
        for record in records:
            ids = select_candidates(record, tools, hard or {}, candidate_count, seed,
                                    getattr(self, 'selection_overrides', None),
                                    getattr(self, 'selection_protected', None))
            gold = ids.index(record['selected']); tool = tools[record['selected']]
            mask = record['masks']
            reading = getattr(self, 'topk_reader', False) and bool(mask['selection'])
            rows, memories = self.compile([tools[k] for k in ids], memory_indices=list(range(len(ids))) if reading else [gold])
            memory = memories[gold] if reading else memories[0]
            item = {k: rows.new_zeros(()) for k in losses}
            intent = record.get('next_intent', '') if record.get('intent_supervision') else ''
            if intent:
                item['intent'] = self.nll(self.prefix(record['history'], 'intent'), intent)
            if mask['selection']:
                scores = self.selection_scores(record['history'], intent, rows)
                item['selection'] = F.cross_entropy(scores, self._ids([gold])) * mask['selection']
                if reading:
                    item['reader'] = self.training_reader_loss(record, intent, ids, scores, memories) * mask['selection']
            if mask['arguments']:
                prefix = self.prefix(record['history'], 'arguments', memory=memory, document=tool.registration_document, next_intent=intent)
                item['arguments'] = self.argument_loss(prefix, record['arguments']) * mask['arguments']
            from .schema_curriculum import targets as schema_targets, request as schema_request
            targets = schema_targets(tool)
            offset = (record['id'][1] if isinstance(record['id'][1], int) else 0) + seed
            fact = targets[(offset if information_index is None else information_index) % len(targets)]
            request = schema_request(fact)
            prefix = self.prefix([], 'information', memory=memory, document=tool.registration_document, fact=request)
            item['information'] = self.nll(prefix, compact(fact))
            if stage == 2:
                if mask['control']:
                    item['control'] = self.nll(self.prefix(record['history'], 'control'), CONTROL[record['mode']]) * mask['control']
                repair = record.get('repair')
                if repair:
                    prefix = self.prefix(record['history'], 'repair', memory=memory, document=tool.registration_document,
                        previous=repair['previous'], error=repair['error'])
                    item['repair'] = self.argument_loss(prefix, repair['target'])
                item['repair'] = item['repair'] + self.recovery_loss(record, tools)
                if mask['answer']:
                    item['answer'] = self.nll(self.prefix(record['history'], 'answer'), record['answer']) * mask['answer']
            for key in losses: losses[key].append(item[key])
        result = {k: torch.stack(v).mean() for k,v in losses.items()}
        result['loss'] = result['selection'] + result['arguments'] + .6*result['information'] + result['control'] + 1.0*result['repair'] + result['answer'] + result['intent']
        if 'reader' in result: result['loss'] = result['loss'] + result['reader']
        return result
