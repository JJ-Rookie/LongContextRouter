"""Uniquely recover one run from already bound, interleaved API-call records."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence


class TraceChainError(ValueError):
    """No unique, fully reconciled API-call chain can be established."""


_COUNTERS = ("prompt_tokens", "completion_tokens", "api_calls")
_TOKENS = (*_COUNTERS[:2], "total_tokens", "cache_hit_tokens", "cache_miss_tokens")
_EDGE_ROUNDOFF = Decimal("0.00000002")
_ENDPOINT_ROUNDOFF = Decimal("0.00000001")
_COUNT_PATTERN = r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)"
_COST_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CALL_STATS = re.compile(
    rf"input_tokens=({_COUNT_PATTERN}), output_tokens=({_COUNT_PATTERN}), "
    rf"instance_cost=({_COST_PATTERN}), cost=({_COST_PATTERN})"
)


def _parse_call_cost(message: str) -> dict:
    match = _CALL_STATS.fullmatch(message)
    if match is None:
        raise TraceChainError("malformed per-call cost binding")
    prompt, completion, cumulative_cost, request_cost = match.groups()
    return {"input_tokens": int(prompt.replace(",", "")),
            "output_tokens": int(completion.replace(",", "")),
            "instance_cost": cumulative_cost, "request_cost": request_cost}


def bind_call_records(events, parse_response, parse_stats) -> tuple[dict, ...]:
    """Bind each Response to exactly one following cost and cumulative event.

    Callers prefilter exact instance loggers and provide safe response/stats
    parsers. A binding cannot cross the next Response or change logger identity.
    """
    pending, records = None, ()
    for line, logger, message in events:
        if message.startswith("Response:"):
            if pending is not None:
                raise TraceChainError("incomplete API call binding before next Response")
            pending = {**parse_response(message[len("Response:"):].strip()),
                       "response_line": line, "logger": logger}
        elif message.startswith("input_tokens="):
            if pending is None or "cost_line" in pending or pending["logger"] != logger:
                raise TraceChainError("missing or duplicate per-call cost binding")
            pending = {**pending, **_parse_call_cost(message), "cost_line": line}
        elif message.startswith("total_tokens_sent="):
            if pending is None or "cost_line" not in pending or pending["logger"] != logger:
                raise TraceChainError("missing or duplicate cumulative stats binding")
            record = {**pending, "cumulative": parse_stats(message), "stats_line": line}
            records = (*records, record)
            pending = None
    if pending is not None:
        raise TraceChainError("incomplete API call binding at end of trace")
    return records


def _integer(values: Mapping[str, Any], key: str) -> int:
    if key not in values:
        raise TraceChainError(f"missing {key}")
    value = values[key]
    if type(value) is not int or value < 0:
        raise TraceChainError(f"{key} must be a nonnegative integer")
    return value


def _cost(values: Mapping[str, Any], key: str) -> Decimal:
    if key not in values:
        raise TraceChainError(f"missing {key}")
    try:
        cost = Decimal(str(values[key]))
    except (InvalidOperation, ValueError):
        raise TraceChainError(f"invalid {key} cost") from None
    if not cost.is_finite() or cost < 0:
        raise TraceChainError(f"invalid {key} cost")
    return cost


def _counter_tuple(values: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(_integer(values, key) for key in _COUNTERS)


def _checked_record(record: Mapping[str, Any], previous_end: int) -> dict:
    lines = tuple(_integer(record, key) for key in ("response_line", "cost_line", "stats_line"))
    if not previous_end < lines[0] < lines[1] < lines[2]:
        raise TraceChainError("API call binding lines overlap or are out of order")
    identifier = record.get("response_id")
    if not isinstance(identifier, str) or not identifier or len(identifier) > 256:
        raise TraceChainError("missing or invalid response id")
    usage, cumulative = record.get("usage"), record.get("cumulative")
    if not isinstance(usage, Mapping) or not isinstance(cumulative, Mapping):
        raise TraceChainError("missing usage or cumulative counters")
    counts = {key: _integer(usage, key) for key in _TOKENS}
    if counts["cache_hit_tokens"] + counts["cache_miss_tokens"] != counts["prompt_tokens"]:
        raise TraceChainError("cache usage token mismatch")
    if counts["prompt_tokens"] + counts["completion_tokens"] != counts["total_tokens"]:
        raise TraceChainError("total usage token mismatch")
    if _integer(usage, "api_calls") != 1:
        raise TraceChainError("per-request usage must contain one API call")
    if (_integer(record, "input_tokens") != counts["prompt_tokens"]
            or _integer(record, "output_tokens") != counts["completion_tokens"]):
        raise TraceChainError("logged input/output tokens disagree with Response usage")
    counters = _counter_tuple(cumulative)
    if counters[0] < counts["prompt_tokens"] or counters[1] < counts["completion_tokens"] or counters[2] < 1:
        raise TraceChainError("cumulative counters are smaller than this request")
    return {**record, "_counters": counters, "_request_cost": _cost(record, "request_cost"),
            "_instance_cost": _cost(record, "instance_cost")}


def _checked_records(records: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    checked = ()
    previous_end = 0
    for record in records:
        item = _checked_record(record, previous_end)
        checked = (*checked, item)
        previous_end = item["stats_line"]
    return checked


def _candidate_paths(records: tuple[dict, ...]) -> tuple:
    paths, by_counter = (), {}
    for index, item in enumerate(records):
        prompt, completion, calls = item["_counters"]
        usage = item["usage"]
        previous = (prompt - usage["prompt_tokens"], completion - usage["completion_tokens"], calls - 1)
        options = ()
        if previous == (0, 0, 0) and abs(item["_instance_cost"] - item["_request_cost"]) <= _EDGE_ROUNDOFF:
            options = ((index,),)
        for predecessor in by_counter.get(previous, ()):
            cost_delta = records[predecessor]["_instance_cost"] + item["_request_cost"] - item["_instance_cost"]
            if abs(cost_delta) <= _EDGE_ROUNDOFF:
                options = (options + tuple(path + (index,) for path in paths[predecessor]))[:2]
        paths = (*paths, options)
        key = item["_counters"]
        by_counter = {**by_counter, key: (*by_counter.get(key, ()), index)}
    return paths


def _unique_indices(records: tuple, expected: Mapping[str, Any]) -> tuple[int, ...]:
    target = _counter_tuple(expected)
    expected_cost = _cost(expected, "instance_cost")
    if target[2] < 1:
        raise TraceChainError("chain recovery requires a positive expected API-call count")
    paths = _candidate_paths(records)
    endpoints = tuple(
        path for index, item in enumerate(records)
        if item["_counters"] == target
        and abs(item["_instance_cost"] - expected_cost) <= _ENDPOINT_ROUNDOFF
        for path in paths[index]
    )
    if not endpoints:
        raise TraceChainError("no matching cumulative usage chain")
    if len(endpoints) != 1:
        raise TraceChainError("ambiguous cumulative usage chains")
    return endpoints[0]


def select_unique_chain(
    records: Sequence[Mapping[str, Any]], expected_stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exactly one increasing call/token/cost-conserving path to traj.

    Records must already bind Response -> per-call input/output/instance_cost ->
    cumulative token/API stats. GLOBAL_STATS.total_cost is deliberately unused.
    Printed costs have eight decimal places; only adjacent rounding tolerance is
    used for edges. Summed printed request cost is reported, not an exact bill.
    """
    checked = _checked_records(records)
    indices = _unique_indices(checked, expected_stats)
    identifiers = [checked[index]["response_id"] for index in indices]
    if len(set(identifiers)) != len(identifiers):
        raise TraceChainError("duplicate response ids in the selected API chain")
    identifier_bytes = json.dumps(identifiers, separators=(",", ":")).encode("utf-8")
    return {
        "method": "unique_cumulative_stats_chain", "chain_count": 1,
        "selected_records": [dict(records[index]) for index in indices],
        "excluded_records": [dict(item) for index, item in enumerate(records) if index not in indices],
        "selected_response_ids_sha256": hashlib.sha256(identifier_bytes).hexdigest(),
        "printed_request_cost_sum": float(sum(checked[index]["_request_cost"] for index in indices)),
        "final_instance_cost": float(checked[indices[-1]]["_instance_cost"]),
        "expected_stats": dict(expected_stats),
    }
