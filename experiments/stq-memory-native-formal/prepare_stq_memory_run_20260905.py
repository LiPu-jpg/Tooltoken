"""Add original STQ argument supervision to the existing native data contract.

No test split, model core, tool descriptions, or benchmark labels are rewritten.
The one STQ entity argument is represented by the pre-existing `entity` schema.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_rows(path, values):
    with path.open("x", encoding="utf-8") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-prepared", type=Path, required=True)
    ap.add_argument("--raw-data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    tools = rows(args.native_prepared / "tools.jsonl")
    retrieval = rows(args.native_prepared / "retrieval.jsonl")
    by_key = {(r["split"], r["endpoint_name"]): r for r in tools}
    by_hash = {r["query_hash"]: r for r in retrieval}
    readback = []
    sources = []
    for split, filename in [
        ("train", "SimpleToolQuestions/train.jsonl"),
        ("validation", "SimpleToolQuestions/dev.jsonl"),
        ("test", "SimpleToolQuestions_unseen/test_unseen_837.jsonl"),
    ]:
        source = args.raw_data / filename
        sources.append(source)
        for raw in rows(source):
            if split == "test":
                query = raw["question"].strip()
                tool_name = raw["tool"].strip().strip("<>")
                value = raw["param"]
            else:
                assert len(raw["tool_calling"]) == 1, "STQ single-call contract changed"
                call = raw["tool_calling"][0]
                assert len(call["call_param"]) == 1
                query = raw["text"][:int(raw["start_str_idx"][0])].strip()
                tool_name = call["call_tool"].strip().strip("<>")
                value = call["call_param"][0]
            assert isinstance(value, str), "Do not coerce non-string gold arguments"
            qh = sha(f"stq:{split}:{raw['case_idx']}:{query}")
            tool = by_key[(split, tool_name)]
            episode = by_hash[qh]
            assert episode["query"] == query
            assert episode["tool_identity_hashes"] == [tool["identity_hash"]]
            readback.append({
                "source_id": f"stq:{split}:{raw['case_idx']}",
                "query_hash": qh, "query_id": str(raw["case_idx"]),
                "query": query, "split": split,
                "tool_identity_hash": tool["identity_hash"],
                "arguments": {"entity": value}, "call_index": 0, "call_count": 1,
            })
    counts = Counter(r["split"] for r in readback)
    assert counts == {"train": 10483, "validation": 1707, "test": 1066}
    assert len({r["query_hash"] for r in readback}) == len(readback)
    norm = lambda x: " ".join(x.lower().split())
    tq = {norm(r["query"]) for r in readback if r["split"] == "train"}
    eq = {norm(r["query"]) for r in readback if r["split"] == "test"}
    assert not tq & eq, "Train/test query overlap: stop, do not silently alter split"
    train_names = {r["endpoint_name"] for r in tools if r["split"] == "train"}
    test_names = {r["endpoint_name"] for r in tools if r["split"] == "test"}
    assert len(train_names) == 999 and len(test_names) == 837
    assert not train_names & test_names
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ["tools.jsonl", "retrieval.jsonl", "split_manifest.json"]:
        original = args.native_prepared / name
        (args.output_dir / name).write_bytes(original.read_bytes())
        sources.append(original)
    write_rows(args.output_dir / "readback.jsonl", readback)
    manifest = {
        "kind": "stq-native-meta-readback-source-supervision-v1",
        "counts": dict(counts), "train_tools": 999, "test_tools": 837,
        "train_test_semantic_tool_overlap": 0, "train_test_query_overlap": 0,
        "argument_contract": "One original call_param value / test param mapped to entity",
        "documents_unchanged": True, "retrieval_split_unchanged": True,
        "validation_note": "Official dev shares 999 seen tool semantics; IDs remain split-disjoint",
        "source_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        "output_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in args.output_dir.iterdir() if f.is_file()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
