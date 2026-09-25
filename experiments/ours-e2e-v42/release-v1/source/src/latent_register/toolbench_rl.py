"""Bounded multi-turn, candidate-conditional group-relative policy optimization.

The frozen rollout policy proposes top-K from the complete TRAIN registry. The
learned categorical action is conditional on that recorded proposal, not a
differentiable claim about full-registry top-K. Environment observations are
conditions only. Token actions include EOS. No gold actions enter rollouts.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .toolbench_curriculum import render_parts
from .toolbench_curriculum_data import CONTROL, clean_history
from .toolbench_data import FINISH, compact, strict_json


def stable_prefix(agent, event, memory, *, request=False, document=""):
    # One embedding-module invocation for every event type: stable ZeRO trace.
    intent = event.get('next_intent', '') if request else ''
    parts = render_parts(agent.tokenizer, event['history'], ('intent' if intent else 'control') if request else event['kind'],
        condition=agent.condition, document=document, previous=event.get('previous', ''), error=event.get('error', ''), fact=event.get("fact", "capability"), history_format=agent.history_format,
        next_intent=event.get("next_intent", "") if not request and event["kind"] == "arguments" else "")
    if request:
        parts[0] += (agent.tokenizer.encode(intent, add_special_tokens=False) + [agent.tokenizer.eos_token_id] if intent else agent.tokenizer.encode(CONTROL['tool'], add_special_tokens=False))
    ids = [i for part in parts for i in part]
    embedded = agent.backbone.get_input_embeddings()(agent._ids(ids))
    if len(parts) == 2:
        embedded = torch.cat([embedded[:len(parts[0])], memory.to(embedded.dtype), embedded[len(parts[0]):]])
    prefix = embedded.unsqueeze(0)
    if prefix.shape[1] > agent.limits.context:
        raise ValueError('context_limit')
    return prefix


def token_logprobs(agent, prefix, ids):
    if not ids or prefix.shape[1] + len(ids) > agent.limits.context:
        raise ValueError('empty_or_overflow_action')
    target = agent._ids(ids)
    embeds = agent.backbone.get_input_embeddings()(target).unsqueeze(0)
    hidden = agent._hidden(inputs_embeds=torch.cat([prefix, embeds], 1), use_cache=False).last_hidden_state
    logits = agent.backbone.get_output_embeddings()(hidden[:, prefix.shape[1]-1:prefix.shape[1]-1+len(ids)]).float()[0]
    return -F.cross_entropy(logits, target, reduction='none')


@torch.no_grad()
def sample_text(agent, prefix, limit, generator):
    # Untruncated categorical sampling at temperature 1: rescorable full support.
    ids, logps = [], []
    cache = None
    inputs = prefix
    for _ in range(min(limit, max(0, agent.limits.context-prefix.shape[1]))):
        output = agent._hidden(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
        logp = F.log_softmax(agent.backbone.get_output_embeddings()(output.last_hidden_state[:, -1]).float()[0], -1)
        token = int(torch.multinomial(logp.exp(), 1, generator=generator))
        ids.append(token); logps.append(float(logp[token]))
        if token == agent.tokenizer.eos_token_id:
            break
        cache = output.past_key_values
        inputs = agent.backbone.get_input_embeddings()(agent._ids([token])).unsqueeze(0)
    if not ids:
        raise ValueError('context_limit')
    stopped = ids[-1] == agent.tokenizer.eos_token_id
    text = agent.tokenizer.decode(ids[:-1] if stopped else ids, skip_special_tokens=False)
    return dict(tokens=ids, old_logps=logps, text=text, stopped=stopped)


def clipped_loss(new, old, advantage, epsilon=.2):
    if new.shape != old.shape or not torch.isfinite(new).all() or not torch.isfinite(old).all():
        raise ValueError('Invalid policy log probabilities')
    ratio = torch.exp(new-old)
    if not torch.isfinite(ratio).all():
        raise ValueError('Nonfinite importance ratio')
    adv = new.new_tensor(advantage)
    return -torch.minimum(ratio*adv, ratio.clamp(1-epsilon, 1+epsilon)*adv).mean()


class PolicyLearner(nn.Module):
    def __init__(self, agent):
        super().__init__(); self.agent = agent

    def forward(self, event, tools):
        # Same sequence of shared modules on all ranks, including masked paths.
        ids = event['candidates']
        if len(set(ids)) != len(ids) or event['selected'] not in ids or FINISH in ids:
            raise ValueError('Invalid recorded action support')
        rows, memories = self.agent.compile([tools[i] for i in ids], memory_indices=[ids.index(event['selected'])])
        memory = memories[0]
        qprefix = stable_prefix(self.agent, event, memory, request=True, document=tools[event['selected']].registration_document)
        q = self.agent._hidden(inputs_embeds=qprefix, use_cache=False).last_hidden_state[:, -1].float()
        selection = F.log_softmax((q @ rows.float().T)[0], -1)[ids.index(event['selected'])].reshape(1)
        text_event = event if event['kind'] != 'selection' else {**event, 'kind': 'control'}
        prefix = stable_prefix(self.agent, text_event, memory, document=tools[event['selected']].registration_document)
        tokens = event['tokens'] if event['kind'] != 'selection' else [self.agent.tokenizer.eos_token_id]
        text = token_logprobs(self.agent, prefix, tokens)
        new = selection if event['kind'] == 'selection' else text
        # Explicit zero dependency for all compiler branches; actual nonzero
        # RL gradients are tested separately on selection and memory events.
        zero = rows.sum()*0 + memories.sum()*0 + selection.sum()*0 + text.sum()*0
        if event.get('supervised'):
            loss = -new.mean() * event['weight']
        else:
            old = new.new_tensor(event['old_logps'])
            loss = clipped_loss(new, old, event['advantage']) * event['weight']
        diagnostics = {'kind': event['kind'], 'supervised': bool(event.get('supervised')),
                       'advantage': float(event.get('advantage', 0)),
                       'trajectory_advantage': float(event.get('trajectory_advantage', event.get('advantage', 0))),
                       'constraint_tags': event.get('local_credit', {}).get('tags', [])}
        if not event.get('supervised'):
            log_ratio = (new-old).detach(); ratio = log_ratio.exp()
            diagnostics.update(mean_log_ratio=float(log_ratio.mean()),
                max_abs_log_ratio=float(log_ratio.abs().max()),
                approximate_kl=float((ratio-1-log_ratio).mean()),
                ratio_outside_clip_fraction=float(((ratio < .8) | (ratio > 1.2)).float().mean()))
        return {'loss': loss+zero, 'logps': new, 'diagnostics': diagnostics}


@torch.no_grad()
def collect_episode(agent, tools, query, execute, *, seed, candidate_count=16,
                    max_decisions=6, max_calls=4, control_tokens=32, argument_tokens=256, answer_tokens=1024, repeat_contracts=None):
    identities = sorted(i for i in tools if i != FINISH)
    if len(identities) < candidate_count:
        raise ValueError('Registry smaller than fixed support')
    generator = torch.Generator(device=agent.device).manual_seed(seed)
    history = [{'type': 'user', 'content': query}]
    events, calls = [], []
    status, answer, repair, failures = 'decision_limit', '', None, 0
    default = identities[:candidate_count]
    def event(kind, support, selected, **kwargs):
        return dict(kind=kind, history=copy.deepcopy(clean_history(history)),
                    candidates=list(support), selected=selected, **kwargs)
    def generate(e, budget):
        memory = agent._registry[e['selected']][2]
        generated = sample_text(agent, stable_prefix(agent, e, memory, document=tools[e['selected']].registration_document), budget, generator)
        e.update(generated); events.append(e)
        return generated
    for decision in range(max_decisions):
        try:
            if repair is None:
                control = generate(event('control', default, default[0]), control_tokens)
                mode = control['text'].strip()
                if not control['stopped'] or mode not in CONTROL.values():
                    status = 'invalid_control'; break
                if mode == CONTROL['give_up']:
                    status = 'give_up'; break
                if mode == CONTROL['final']:
                    final = generate(event('answer', default, default[0]), answer_tokens)
                    answer = final['text']; status = 'final' if final['stopped'] and answer.strip() else 'invalid_final'; break
                if len(calls) >= max_calls:
                    status = 'call_limit'; break
                # q is fixed before candidates; proposal never receives gold.
                all_rows = torch.stack([agent._registry[i][1] for i in identities])
                intent = generate(event('intent', default, default[0]), 96)
                if not intent['stopped'] or not intent['text'].strip():
                    status = 'invalid_intent'; break
                scores = agent.selection_scores(clean_history(history), intent['text'], all_rows)[0]
                indices = scores.argsort(descending=True, stable=True)[:candidate_count]
                support = [identities[i] for i in indices.tolist()]
                logp = F.log_softmax(scores[indices].float(), -1)
                index = int(torch.multinomial(logp.exp(), 1, generator=generator))
                selected = support[index]
                events.append(event('selection', support, selected, tokens=[], old_logps=[float(logp[index])], next_intent=intent['text']))
                args = generate(event('arguments', support, selected, next_intent=intent['text']), argument_tokens)
            else:
                selected, support, previous, error = repair
                args = generate(event('repair', support, selected, previous=previous, error=error), argument_tokens)
                if args['stopped'] and args['text'].strip() == '<reselect>':
                    history.append({'type': 'recovery', 'api_identity': selected, 'previous_arguments': previous, 'error': error, 'action': '<reselect>'})
                    repair = None
                    continue
            try:
                if not args['stopped']:
                    raise ValueError('incomplete_arguments')
                arguments = tools[selected].validate_arguments(strict_json(args['text']))
            except ValueError as exc:
                failures += 1
                if failures > 2:
                    status = 'validation_limit'; break
                repair = (selected, support, args['text'], str(exc)); continue
            repair = None
            history.append({'type': 'call', 'thought': CONTROL['tool'], 'api_identity': selected, 'arguments': arguments})
            result = execute(selected, arguments)
            current_call={'api_identity': selected, 'arguments': arguments, 'result': result.content, 'receipt': result.metadata}
            from .action_credit import certify_repeat
            contract=(repeat_contracts or {}).get(selected,{})
            proof=certify_repeat(calls[-1] if calls else None,current_call,
                stable_read_contract=contract.get('stable_read_contract'),
                freshness_requested=contract.get('freshness_requested'))
            proof['call_index']=len(calls)
            # generate() stored a distinct event dict; attach evidence only to
            # the actual executed argument/repair event, never to a control token.
            events[-1]['redundancy_evidence']=proof
            calls.append(current_call)
            history.append({'type': 'observation', 'api_identity': selected, 'content': result.content})
            if result.metadata.get('transport_status') != 'ok' or result.content.get('error') == 'Failed to generate fake response':
                status = 'infrastructure_failure'; break
        except ValueError as exc:
            if str(exc) not in {'context_limit', 'empty_or_overflow_action'}:
                raise  # Do not hide implementation errors as model failures.
            status = 'context_limit'; break
    return dict(query=query, seed=seed, history=history, events=events, calls=calls,
                status=status, answer=answer, validation_failures=failures,
                eligible=status != 'infrastructure_failure')


def group_advantages(episodes):
    if len(episodes) < 2 or any(not e['eligible'] for e in episodes):
        return None  # An infrastructure-broken group is never a losing policy.
    rewards = [float(e['reward']) for e in episodes]
    if not all(math.isfinite(r) for r in rewards):
        raise ValueError('Nonfinite reward')
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/max(std, 1e-6) for r in rewards]


def evidence_reward(episode, verdict):
    """Model-judged simulated outcomes, not verified real-world correctness."""
    if not episode['eligible']:
        return None
    # Terminal-answer reward: progress shaping must be a separate measured objective.
    if episode['status'] != 'final' or not (episode.get('answer') or '').strip():
        return 0.
    observations = [c['result'] for c in episode['calls']]
    citations = verdict.get('citations', [])
    supported = bool(citations) and all(isinstance(c, dict) and type(c.get('observation')) is int
        and 0 <= c['observation'] < len(observations) and isinstance(c.get('quote'), str)
        and len(c['quote'].strip()) >= 2 and observations[c['observation']].get('error') == ''
        and c['quote'] in compact(observations[c['observation']].get('response')) for c in citations)
    coverage = verdict.get('coverage')
    if type(coverage) not in (int, float) or not math.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError('Malformed judge coverage')
    if type(verdict.get('complete')) is not bool or type(verdict.get('unsupported')) is not bool:
        raise ValueError('Malformed judge decision')
    if verdict['complete'] and coverage != 1:
        raise ValueError('Inconsistent judge completion/coverage/support')
    usable = supported and not verdict['unsupported']
    outcome = (1.0 if episode['status'] == 'final' and verdict['complete'] and coverage == 1 else .2*coverage) if usable else 0.
    signatures = [compact([c['api_identity'], c['arguments']]) for c in episode['calls']]
    # Ordinary execution cost is separate from redundant-action credit. Identical
    # signatures alone do not establish redundancy (refresh/retry may be needed).
    # Certified repeats are constrained in local_credit, not penalized twice here.
    cost = min(.15, .01*len(signatures)+.01*episode['validation_failures'])
    return outcome*(1-cost) if outcome > 0 else 0.
