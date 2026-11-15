from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from .cached_billing_chain import TraceChainError, bind_call_records, select_unique_chain


class CachedBillingError(ValueError):
    pass


_LOG_LINE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - "
    r"(?:DEBUG|INFO|WARNING|ERROR|CRITICAL|TRACE) - (swea-(?:lm|agent)(?:-[^\s]+)?) - (.*)$"
)
_COUNT = r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_STATS = re.compile(
    rf"total_tokens_sent=({_COUNT}), total_tokens_received=({_COUNT}), "
    rf"total_cost={_NUMBER}, total_api_calls=({_COUNT})"
)
_TOKEN_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cache_hit_tokens", "cache_miss_tokens",
)
_COUNTER_KEYS = ("prompt_tokens", "completion_tokens", "api_calls")


def _keywords(node: ast.AST, expected_name: str) -> dict[str, ast.AST]:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == expected_name and not node.args):
        raise CachedBillingError(f"expected {expected_name} keyword object")
    if any(item.arg is None for item in node.keywords):
        raise CachedBillingError("expanded keyword arguments are not valid usage")
    fields = {item.arg: item.value for item in node.keywords}
    if len(fields) != len(node.keywords):
        raise CachedBillingError("duplicate keyword fields")
    return fields


def _integer_value(values: Mapping[str, Any], key: str) -> int:
    if key not in values:
        raise CachedBillingError(f"missing {key}")
    value = values[key]
    if type(value) is not int or value < 0:
        raise CachedBillingError(f"{key} must be a nonnegative integer")
    return value


def _integer_node(fields: Mapping[str, ast.AST], key: str) -> int:
    if key not in fields:
        raise CachedBillingError(f"missing {key}")
    node = fields[key]
    if not isinstance(node, ast.Constant):
        raise CachedBillingError(f"{key} must be a nonnegative integer literal")
    return _integer_value({key: node.value}, key)


def _validate_tokens(usage: Mapping[str, Any]) -> None:
    counts = {key: _integer_value(usage, key) for key in _TOKEN_KEYS}
    if counts["cache_hit_tokens"] + counts["cache_miss_tokens"] != counts["prompt_tokens"]:
        raise CachedBillingError("cache hit + miss versus prompt token mismatch")
    if counts["prompt_tokens"] + counts["completion_tokens"] != counts["total_tokens"]:
        raise CachedBillingError("prompt + completion versus total token mismatch")


def _check_alias(fields: Mapping[str, ast.AST], hit_tokens: int) -> None:
    details = fields.get("prompt_tokens_details")
    if details is None or (isinstance(details, ast.Constant) and details.value is None):
        return
    detail_fields = _keywords(details, "PromptTokensDetailsWrapper")
    if "cached_tokens" not in detail_fields:
        return
    alias = detail_fields["cached_tokens"]
    if isinstance(alias, ast.Constant) and alias.value is None:
        return
    if _integer_node(detail_fields, "cached_tokens") != hit_tokens:
        raise CachedBillingError("cached_tokens versus native cache hit token mismatch")


def _response_keywords(expression: str) -> dict:
    try:
        parsed = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        raise CachedBillingError("invalid ModelResponse syntax") from None
    return _keywords(parsed, "ModelResponse")


def _parse_response(expression: str) -> dict[str, int]:
    response = _response_keywords(expression)
    if "usage" not in response:
        raise CachedBillingError("Response is missing usage")
    fields = _keywords(response["usage"], "Usage")
    usage = {
        "prompt_tokens": _integer_node(fields, "prompt_tokens"),
        "completion_tokens": _integer_node(fields, "completion_tokens"),
        "total_tokens": _integer_node(fields, "total_tokens"),
        "cache_hit_tokens": _integer_node(fields, "prompt_cache_hit_tokens"),
        "cache_miss_tokens": _integer_node(fields, "prompt_cache_miss_tokens"),
        "api_calls": 1,
    }
    _validate_tokens(usage)
    _check_alias(fields, usage["cache_hit_tokens"])
    return usage


def _response_record(expression: str) -> dict:
    usage = _parse_response(expression)
    identifier = _response_keywords(expression).get("id")
    if not isinstance(identifier, ast.Constant) or not isinstance(identifier.value, str):
        raise CachedBillingError("chain recovery requires an explicit response id")
    return {"usage": usage, "response_id": identifier.value}


def _parse_stats(message: str) -> dict[str, int]:
    match = _STATS.fullmatch(message)
    if match is None:
        raise CachedBillingError("malformed cumulative trace stats")
    counts = (int(value.replace(",", "")) for value in match.groups())
    return dict(zip(_COUNTER_KEYS, counts))


def reconcile_usage(usage: Mapping[str, Any], stats: Mapping[str, Any]) -> None:
    for key in _COUNTER_KEYS:
        if _integer_value(usage, key) != _integer_value(stats, key):
            raise CachedBillingError(f"cumulative {key} mismatch")


def _instance_identity(path: Path, expected: str | None) -> str:
    if expected is None:
        if not path.name.endswith(".trace.log"):
            raise CachedBillingError("cannot infer instance identity from trace filename")
        expected = path.name[:-len(".trace.log")]
    if not isinstance(expected, str) or not re.fullmatch(r"[^\s/\\]+", expected):
        raise CachedBillingError("instance identity must be a nonempty artifact identifier")
    return expected


def _trace_events(path: Path):
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                raise CachedBillingError(f"{path}:{line_number}: invalid UTF-8") from None
            match = _LOG_LINE.fullmatch(line)
            if match is not None:
                yield line_number, match.group(1), match.group(2)


def _foreign_event(records: dict, logger: str, message: str, line: int) -> dict:
    previous = records.get(logger, {
        "logger": logger, "response_count": 0, "stats_count": 0,
        "line_count": 0, "first_line": line,
    })
    current = {**previous,
               "response_count": previous["response_count"] + int(message.startswith("Response:")),
               "stats_count": previous["stats_count"] + int(message.startswith("total_tokens_sent=")),
               "line_count": previous["line_count"] + 1, "last_line": line}
    return {**records, logger: current}


def _run_start(path: Path, identity: str, logger: str, message: str) -> bool:
    prefix = "Trajectory will be saved to "
    if logger not in {"swea-agent", f"swea-agent-{identity}"} or not message.startswith(prefix):
        return False
    destination = Path(message[len(prefix):])
    expected = path.with_name(f"{identity}.traj").absolute()
    if not destination.is_absolute():
        return False
    if destination == expected:
        return True
    anchor = "sweagent_results"
    if any(item.parts.count(anchor) != 1 for item in (destination, expected)):
        return False
    source_parts = destination.parts[destination.parts.index(anchor):]
    expected_parts = expected.parts[expected.parts.index(anchor):]
    return len(expected_parts) == 6 and source_parts == expected_parts


def _empty_run(
    start_line: int = 1, explicit_start: bool = False, trajectory_path: str | None = None,
) -> dict:
    return {"start_line": start_line, "explicit_start": explicit_start,
            "trajectory_path": trajectory_path,
            "usage": {key: 0 for key in (*_TOKEN_KEYS, "api_calls")}, "trace_stats": None}


def _collect_runs(path: Path, identity: str) -> tuple[tuple, dict]:
    accepted = {"swea-lm", f"swea-lm-{identity}"}
    runs, current, foreign = (), _empty_run(), {}
    for line_number, logger, message in _trace_events(path):
        if logger.startswith("swea-agent"):
            if _run_start(path, identity, logger, message):
                if current["explicit_start"] or current["usage"]["api_calls"] or current["trace_stats"] is not None:
                    runs = (*runs, {**current, "end_line": line_number - 1})
                current = _empty_run(line_number, explicit_start=True,
                                     trajectory_path=message[len("Trajectory will be saved to "):])
            continue
        if logger not in accepted:
            foreign = _foreign_event(foreign, logger, message, line_number)
            continue
        try:
            if message.startswith("Response:"):
                usage = _parse_response(message[len("Response:"):].strip())
                totals = {key: value + usage[key] for key, value in current["usage"].items()}
                current = {**current, "usage": totals}
            elif message.startswith("total_tokens_sent="):
                current = {**current, "trace_stats": _parse_stats(message)}
        except CachedBillingError as error:
            raise CachedBillingError(f"{path}:{line_number}: {error}") from None
    return (*runs, current), foreign


def _validate_runs(runs: tuple) -> None:
    for run in runs:
        if run["trace_stats"] is not None:
            reconcile_usage(run["usage"], run["trace_stats"])
        elif len(runs) > 1:
            raise CachedBillingError("explicit run is missing cumulative trace stats")


def _recover_interleaved(path, identity, runs, foreign, expected) -> dict:
    accepted = {"swea-lm", f"swea-lm-{identity}"}
    events = (event for event in _trace_events(path) if event[1] in accepted)
    try:
        records = bind_call_records(events, _response_record, _parse_stats)
        recovered = select_unique_chain(records, expected)
    except TraceChainError as error:
        raise CachedBillingError(f"{path}: {error}") from None
    selected = recovered["selected_records"]
    totals = {key: sum(record["usage"][key] for record in selected)
              for key in (*_TOKEN_KEYS, "api_calls")}
    reconcile_usage(totals, expected)
    recovery = {**recovered, "source_trace": str(path),
                "fallback_reason": "explicit_runs_failed_sequential_counter_validation",
                "run_start_markers": [
                    {key: run[key] for key in ("start_line", "trajectory_path")}
                    for run in runs if run["explicit_start"]]}
    return {**totals, "status": "parsed",
            "trace_stats": selected[-1]["cumulative"], "selection": {
                "expected_instance_id": identity,
                "foreign_loggers": [foreign[key] for key in sorted(foreign)],
                "chain_recovery": recovery}}


def parse_trace_usage(
    path: str | Path, *, expected_instance_id: str | None = None,
    expected_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace_path = Path(path)
    identity = _instance_identity(trace_path, expected_instance_id)
    runs, foreign = _collect_runs(trace_path, identity)
    try:
        _validate_runs(runs)
    except CachedBillingError:
        if expected_stats is None or sum(run["explicit_start"] for run in runs) < 2:
            raise
        return _recover_interleaved(trace_path, identity, runs, foreign, expected_stats)
    selected = runs[-1]
    totals, final_stats = selected["usage"], selected["trace_stats"]
    if expected_stats is not None:
        reconcile_usage(totals, expected_stats)
    selection = {"expected_instance_id": identity,
                 "foreign_loggers": [foreign[key] for key in sorted(foreign)]}
    if len(runs) > 1:
        selection = {**selection, "run_selection": {
            "boundary": "exact_trajectory_start", "run_count": len(runs),
            "selected_run_index": len(runs) - 1,
            "selected_start_line": selected["start_line"],
            "excluded_runs": list(runs[:-1]),
        }}
    status = "parsed" if totals["api_calls"] else (
        "zero_api_calls" if final_stats is not None else "no_response")
    return {**totals, "status": status,
            "trace_stats": final_stats, "selection": selection}


def _price(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise CachedBillingError(f"{name} must be a finite nonnegative price")
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise CachedBillingError(f"{name} must be a finite nonnegative price") from None
    if not price.is_finite() or price < 0:
        raise CachedBillingError(f"{name} must be a finite nonnegative price")
    return price


def calculate_cached_cost(
    usage: Mapping[str, Any], input_per_million: Any,
    cached_input_per_million: Any, output_per_million: Any,
) -> float:
    _validate_tokens(usage)
    uncached = _price(input_per_million, "input_per_million")
    cached = _price(cached_input_per_million, "cached_input_per_million")
    output = _price(output_per_million, "output_per_million")
    total = (usage["cache_miss_tokens"] * uncached
             + usage["cache_hit_tokens"] * cached
             + usage["completion_tokens"] * output) / Decimal(1_000_000)
    result = float(total)
    if not math.isfinite(result):
        raise CachedBillingError("calculated cost exceeds finite JSON number range")
    return result
