import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("latebound_train", ROOT / "scripts" / "train.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_epoch_schedule_consumes_each_epoch_without_implicit_repeats():
    groups = MODULE._epoch_step_groups(
        example_count=5,
        batch_size=2,
        gradient_accumulation=2,
        epochs=2,
        seed=17,
    )
    assert len(groups) == 4
    for epoch in (0, 1):
        seen = [index for group_epoch, batches in groups if group_epoch == epoch for batch in batches for index in batch]
        assert sorted(seen) == list(range(5))


def test_step_limit_is_a_prefix_of_the_epoch_schedule():
    groups = MODULE._epoch_step_groups(5, 2, 2, 2, 17)
    assert groups[:2] == MODULE._epoch_step_groups(5, 2, 2, 1, 17)[:2]


def test_distributed_schedule_pads_only_the_final_global_batch():
    rank_groups = MODULE._distributed_epoch_step_groups(5, 2, 1, 1, 17, 2, 0)
    other_groups = MODULE._distributed_epoch_step_groups(5, 2, 1, 1, 17, 2, 1)
    assert len(rank_groups) == len(other_groups) == 2
    flattened = [index for _, batches in rank_groups for batch in batches for index in batch]
    other_flattened = [index for _, batches in other_groups for batch in batches for index in batch]
    assert flattened.count(None) == 1
    assert other_flattened.count(None) == 2
    assert sorted(index for index in flattened + other_flattened if index is not None) == [0, 1, 2, 3, 4]
