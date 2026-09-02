import json

from latebound_sequence_sft.data import build_examples, load_corpus


def test_sequence_conversion_keeps_exact_doc_and_no_cartesian_expansion(tmp_path):
    corpus_path = tmp_path / "corpus.tsv"
    corpus_path.write_text(
        "\tdocid\tdocument_content\n"
        "0\ta\t{\"tool_name\": \"T\", \"api_name\": \"A\", \"schema\": 1}\n"
        "1\tb\t{\"tool_name\": \"T\", \"api_name\": \"A\", \"schema\": 2}\n"
        "2\tc\t{\"tool_name\": \"U\", \"api_name\": \"B\"}\n",
        encoding="utf-8",
    )
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "choose a"},
                        {"role": "assistant", "content": "<<T&&A>>"},
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "choose b"},
                        {"role": "assistant", "content": "<<U&&B>>"},
                    ]
                },
            ]
        ),
        encoding="utf-8",
    )
    corpus, _ = load_corpus(corpus_path)
    examples, excluded = build_examples(
        retrieval_path, corpus, excluded_queries={"not present"}
    )
    assert excluded == 0
    assert len(examples) == 3
    assert [example.document_id for example in examples] == ["a", "b", "c"]
    assert examples[0].positive_document_count == 2
    assert examples[0].loss_weight == 0.5
    assert examples[2].positive_document_count == 1


def test_evaluation_query_is_removed_before_materialization(tmp_path):
    corpus_path = tmp_path / "corpus.tsv"
    corpus_path.write_text(
        "\tdocid\tdocument_content\n"
        "0\ta\t{\"tool_name\": \"T\", \"api_name\": \"A\"}\n",
        encoding="utf-8",
    )
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "held out"},
                        {"role": "assistant", "content": "<<T&&A>>"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus, _ = load_corpus(corpus_path)
    examples, excluded = build_examples(
        retrieval_path, corpus, excluded_queries={"held out"}
    )
    assert examples == []
    assert excluded == 1
