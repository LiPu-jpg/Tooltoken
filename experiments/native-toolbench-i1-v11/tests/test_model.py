from types import SimpleNamespace

import torch
from torch import nn

from latebound_sequence_sft.model import NativeSequenceSFT, registration_zero_step


class ToyDecoder(nn.Module):
    def __init__(self, vocab_size=10, hidden_size=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **_):
        value = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=self.proj(value))


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyDecoder()
        self.lm_head = nn.Linear(4, 10, bias=False)

    def get_input_embeddings(self):
        return self.model.embed

    def get_output_embeddings(self):
        return self.lm_head


class CountingCompiler(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        self.calls = []

    def output_rows(self, states):
        self.calls.append("output")
        return self.output.expand(len(states), -1)

    def registered_memory(self, states):
        self.calls.append("memory")
        return states.unsqueeze(1).expand(-1, 2, -1)


def _batch():
    input_ids = torch.tensor([[1, 2, 8, 3], [4, 5, 9, 3]])
    labels = torch.tensor([[-100, -100, 8, 3], [-100, -100, 9, 3]])
    attention = torch.ones_like(input_ids)
    documents = {
        "input_ids": torch.tensor([[6, 7], [7, 6]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    return input_ids, attention, labels, documents, torch.tensor([8, 9])


def test_standard_sequence_ce_masks_reserved_rows_and_uses_one_doc_per_row():
    backbone = ToyBackbone()
    compiler = CountingCompiler()
    model = NativeSequenceSFT(backbone, compiler, reserved_token_ids=[8, 9])
    input_ids, attention, labels, documents, physical_ids = _batch()
    loss, diagnostics = model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        document_tokens=documents,
        physical_ids=physical_ids,
    )
    assert torch.isfinite(loss)
    assert diagnostics.document_forward_count == 2
    assert diagnostics.dynamic_row_count == 2
    assert diagnostics.reserved_rows_masked is True
    assert diagnostics.denominator == "ordinary_vocabulary_plus_active_dynamic_rows"
    assert compiler.calls == ["output", "memory"]


def test_duplicate_physical_address_is_rejected():
    backbone = ToyBackbone()
    model = NativeSequenceSFT(backbone, CountingCompiler(), reserved_token_ids=[8, 9])
    input_ids, attention, labels, documents, _ = _batch()
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention,
            labels=labels,
            document_tokens=documents,
            physical_ids=torch.tensor([8, 8]),
        )
    except ValueError as exc:
        assert "bound to multiple documents" in str(exc)
    else:
        raise AssertionError("duplicate physical addresses must fail closed")


def test_repeated_sequence_reuses_one_document_forward_when_binding_matches():
    backbone = ToyBackbone()
    compiler = CountingCompiler()
    model = NativeSequenceSFT(backbone, compiler, reserved_token_ids=[8, 9])
    input_ids, attention, labels, documents, _ = _batch()
    one_document = {key: value[:1] for key, value in documents.items()}
    labels = labels.clone()
    labels[1, 2] = 8
    loss, diagnostics = model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        document_tokens=one_document,
        document_index=torch.tensor([0, 0]),
        physical_ids=torch.tensor([8, 8]),
    )
    assert torch.isfinite(loss)
    assert diagnostics.document_forward_count == 1


def test_registration_has_no_update_path():
    compiler = CountingCompiler()
    before = {name: value.detach().clone() for name, value in compiler.named_parameters()}
    rows, memory = registration_zero_step(compiler, torch.ones(2, 4))
    assert rows.shape == (2, 4)
    assert memory.shape == (2, 2, 4)
    for name, value in compiler.named_parameters():
        assert torch.equal(value, before[name])


def test_sequence_ce_backpropagates_to_backbone_and_compiler():
    backbone = ToyBackbone()
    compiler = CountingCompiler()
    model = NativeSequenceSFT(backbone, compiler, reserved_token_ids=[8, 9])
    input_ids, attention, labels, documents, physical_ids = _batch()
    loss, _ = model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        document_tokens=documents,
        physical_ids=physical_ids,
        loss_weights=torch.tensor([0.5, 1.0]),
    )
    loss.backward()
    assert backbone.model.embed.weight.grad is not None
    assert backbone.lm_head.weight.grad is not None
    assert compiler.output.grad is not None


def test_zero_weight_padding_is_graph_connected_and_finite():
    backbone = ToyBackbone()
    model = NativeSequenceSFT(backbone, CountingCompiler(), reserved_token_ids=[8, 9])
    input_ids, attention, labels, documents, physical_ids = _batch()
    loss, _ = model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        document_tokens=documents,
        physical_ids=physical_ids,
        loss_weights=torch.zeros(2),
    )
    assert torch.isfinite(loss)
    loss.backward()
