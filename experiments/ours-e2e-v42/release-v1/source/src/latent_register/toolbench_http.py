"""Explicit ToolBench / StableToolBench HTTP binding; no default endpoint or key."""
from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .toolbench_agent import ExecutionResult
from .toolbench_data import FINISH, compact, strict_json


def standardize(name: str) -> str:
    # ToolBench evaluation/toolbench/utils.py:standardize/change_name.
    name = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", name)
    name = re.sub(r"_+", "_", name).lower().strip("_")
    if name and name[0].isdigit():
        name = "get_" + name
    return name


def change_name(name: str) -> str:
    if name in {"from", "class", "return", "false", "true", "id", "and"}:
        return "is_" + name
    return name


def wire_bindings(bindings: dict) -> dict:
    result, occupied = {}, {}
    for identity, source in bindings.items():
        if identity == FINISH:
            continue
        if not all(isinstance(source.get(k), str) and source[k].strip()
                   for k in ("category_name", "tool_name", "api_name")):
            raise ValueError("Missing exact ToolBench source binding")
        wire = {"category": source["category_name"],
                "tool_name": standardize(source["tool_name"]),
                "api_name": change_name(standardize(source["api_name"]))}
        if not wire["tool_name"] or not wire["api_name"]:
            raise ValueError("Empty normalized ToolBench binding")
        # The official server canonicalizes category punctuation as well.
        category = source["category_name"].replace(" ", "_").replace(",", "_").replace("/", "_").replace("__", "_")
        key = (category, wire["tool_name"], wire["api_name"])
        if key in occupied:
            raise ValueError("Two exact API identities collide at the executor endpoint")
        occupied[key] = identity
        result[identity] = wire
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ToolBenchHTTPExecutor:
    def __init__(self, *, query_id, config, tool_bindings):
        self.query_id = query_id
        self.bindings = wire_bindings(tool_bindings)
        self.url = config["service_url"]
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("Supply an explicit service URL without inline credentials")
        self.kind = config["backend_kind"]
        if self.kind not in {"toolbench_real", "stabletoolbench_virtual", "loopback_contract_only"}:
            raise ValueError("Unknown executor backend kind")
        if self.kind == "loopback_contract_only" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Contract fixtures must use loopback")
        self.revision = config["backend_revision"]
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("Record the backend version")
        key_name = config.get("toolbench_key_env")
        self.key = os.environ.get(key_name, "") if key_name else ""
        if not self.key and not config.get("allow_empty_toolbench_key", False):
            raise ValueError("Configured ToolBench credential is missing")
        self.timeout = float(config.get("timeout_seconds", 120))
        self.max_bytes = int(config.get("max_response_bytes", 1048576))
        if not 0 < self.timeout <= 300 or not 0 < self.max_bytes <= 16777216:
            raise ValueError("Unbounded executor request")
        handlers = [NoRedirect()]
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            # A local executor must not be routed through an external HTTP proxy.
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def __call__(self, identity, arguments):
        if identity not in self.bindings or not isinstance(arguments, dict):
            raise ValueError("Executor accepts only a bound API and a JSON object")
        payload = {**self.bindings[identity], "tool_input": compact(arguments),
                   "strip": "", "toolbench_key": self.key}
        request = urllib.request.Request(self.url, data=compact(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "toolbench_key": self.key})
        started = time.monotonic()
        receipt = {"backend_kind": self.kind, "backend_revision": self.revision,
                   "api_identity": identity, "wire_binding": self.bindings[identity],
                   "arguments_sha256": hashlib.sha256(compact(arguments).encode()).hexdigest(),
                   "attempts": 1, "http_status": None}
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                receipt["http_status"] = response.status
                raw = response.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                raise ValueError("response_size_limit")
            result = strict_json(raw.decode("utf-8"))
            if not isinstance(result, dict) or not isinstance(result.get("error"), str) or "response" not in result:
                raise ValueError("invalid_response_envelope")
            receipt["transport_status"] = "ok"
            success = result["error"] == ""
        except urllib.error.HTTPError as exc:
            receipt.update(http_status=exc.code, transport_status="http_error")
            result, success = {"error": f"ToolBench HTTP status {exc.code}", "response": ""}, False
        except (urllib.error.URLError, TimeoutError, OSError):
            receipt["transport_status"] = "network_error"
            result, success = {"error": "ToolBench executor network failure", "response": ""}, False
        except (ValueError, UnicodeError):
            receipt["transport_status"] = "response_contract_error"
            result, success = {"error": "ToolBench response failed the transport contract", "response": ""}, False
        receipt["seconds"] = time.monotonic() - started
        # The full envelope, including actual tool errors, reaches the caller.
        return ExecutionResult(result, success, receipt)


def create_executor(*, query_id, config, tool_bindings):
    return ToolBenchHTTPExecutor(query_id=query_id, config=config, tool_bindings=tool_bindings)
