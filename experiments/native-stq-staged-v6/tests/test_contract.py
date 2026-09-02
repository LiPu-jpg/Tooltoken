from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_new_training_tree_contains_no_old_retrieval_switches():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src").rglob("*.py")
    )
    assert "multi_positive_loss" not in source
    assert "negative_count" not in source
    assert "sampled-softmax" not in source


def test_multi_gpu_entry_is_explicitly_gated_and_smoke_reloads_checkpoint():
    train = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    smoke = (ROOT / "slurm" / "l20_smoke.sbatch").read_text(encoding="utf-8")
    ds_config = (ROOT / "slurm" / "ds_z3_micro1_config.json").read_text(encoding="utf-8")
    reload_script = ROOT / "scripts" / "reload_checkpoint.py"
    assert "WORLD_SIZE>1 requires --deepspeed-config" in train
    assert "--expected-world-size" in train
    assert "torch.distributed.run" in smoke
    assert "--deepspeed-config" in smoke
    assert "reload_checkpoint.py" in smoke
    assert "TORCH_EXTENSIONS_DIR" in smoke
    assert "torch.optim.AdamW" in smoke
    assert "adamw_single_process_probe=PASS" in smoke
    assert '"params": {\n      "torch_adam": true' in ds_config
    assert "full_parameter_count" in train
    assert "backbone_requires_grad" in train
    assert "compiler_requires_grad" in train
    assert "optimizer_parameter_group_count" in train
    assert reload_script.is_file()
