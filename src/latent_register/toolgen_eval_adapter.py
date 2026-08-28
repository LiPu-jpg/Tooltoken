from __future__ import annotations

import runpy
import sys
from collections.abc import Callable, MutableMapping
from typing import Any


def install_template(
    registry: MutableMapping[str, Any],
    name: str,
    loader: Callable[[str], Any],
) -> None:
    registry[name] = loader(name)


def register_toolgen_templates() -> None:
    from fastchat.conversation import conv_templates
    from prompts.conversations import get_conv_template

    for name in ("llama-3", "qwen-7b-chat"):
        install_template(conv_templates, name, get_conv_template)


def check_adapter() -> None:
    from fastchat.conversation import get_conv_template

    register_toolgen_templates()
    conversation = get_conv_template("llama-3")
    conversation.append_message(conversation.roles[0], "adapter check")
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    required = (
        "<|start_header_id|>user<|end_header_id|>",
        "adapter check",
        "<|start_header_id|>assistant<|end_header_id|>",
    )
    if not all(value in prompt for value in required):
        raise RuntimeError("ToolGen Llama-3 conversation adapter produced a bad prompt")
    print("ToolGen conversation adapter check passed")


def main() -> None:
    if sys.argv[1:] == ["--check-only"]:
        check_adapter()
        return
    register_toolgen_templates()
    runpy.run_module("evaluation.retrieval.eval_toolgen", run_name="__main__")


if __name__ == "__main__":
    main()
