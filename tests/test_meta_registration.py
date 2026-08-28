import unittest

import torch

from latent_register.episodic_data import (
    BoundRegistryEpisode,
    PreparedRetrievalEpisode,
    PreparedTool,
)
from latent_register.train_meta_registration import (
    _epoch_batches,
    _tokenize_bound_episodes,
    multi_positive_selection_loss,
    selection_statistics,
)


class _BatchEncoding(dict):
    def to(self, _device):
        return self


class _RecordingTokenizer:
    chat_template = None

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, **_kwargs):
        self.calls.append(list(texts))
        width = max(len(text) for text in texts)
        return _BatchEncoding(
            input_ids=torch.ones(len(texts), width, dtype=torch.long),
            attention_mask=torch.ones(len(texts), width, dtype=torch.long),
        )


class MetaRegistrationTests(unittest.TestCase):
    def test_tokenization_applies_the_requested_document_instruction(self) -> None:
        tool = PreparedTool(
            identity_hash="tool",
            group_hash="group",
            split="test",
            document="Tool document.",
            tool_name="tool",
            endpoint_name="endpoint",
            source="test",
        )
        episode = BoundRegistryEpisode(
            query_hash="query",
            query="Use the tool.",
            split="test",
            tools=(tool,),
            slot_indices=(0,),
            positive_positions=(0,),
        )
        tokenizer = _RecordingTokenizer()

        _tokenize_bound_episodes(
            tokenizer,
            [episode],
            max_query_length=64,
            max_document_length=64,
            device=torch.device("cpu"),
            document_instruction="Custom selection view.",
        )

        self.assertEqual(
            tokenizer.calls[1], ["Custom selection view.\nTool document."]
        )

    def test_multi_positive_loss_rewards_any_valid_tool(self) -> None:
        positive_mask = torch.tensor([[True, True, False]])
        good = multi_positive_selection_loss(
            torch.tensor([[0.0, 5.0, -2.0]]), positive_mask
        )
        bad = multi_positive_selection_loss(
            torch.tensor([[-2.0, -3.0, 5.0]]), positive_mask
        )
        self.assertLess(float(good), float(bad))

    def test_selection_statistics_uses_best_positive_rank(self) -> None:
        logits = torch.tensor([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        positive_mask = torch.tensor(
            [[True, True, False], [False, True, False]]
        )
        hits, reciprocal_rank = selection_statistics(logits, positive_mask)
        self.assertEqual(hits, 1)
        self.assertAlmostEqual(reciprocal_rank, 1.5)

    def test_epoch_batches_give_ranks_equal_disjoint_work(self) -> None:
        episodes = [
            PreparedRetrievalEpisode(str(index), "query", "train", ("tool",))
            for index in range(11)
        ]
        rank_zero = list(
            _epoch_batches(
                episodes,
                epoch=0,
                seed=17,
                batch_size=2,
                rank=0,
                world_size=2,
            )
        )
        rank_one = list(
            _epoch_batches(
                episodes,
                epoch=0,
                seed=17,
                batch_size=2,
                rank=1,
                world_size=2,
            )
        )
        self.assertEqual(len(rank_zero), len(rank_one))
        zero_ids = {item.query_hash for batch in rank_zero for item in batch}
        one_ids = {item.query_hash for batch in rank_one for item in batch}
        self.assertFalse(zero_ids.intersection(one_ids))
        self.assertEqual(len(zero_ids | one_ids), 8)


if __name__ == "__main__":
    unittest.main()
