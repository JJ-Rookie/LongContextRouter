"""Fail-closed billing for a SWE-agent exploration prefix.

The full trace is reconciled against trajectory counters and the frozen cost
before any per-step prefix is accepted. A prefix is never prorated from a run.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .cached_billing import (
    CachedBillingError, _LOG_LINE, _parse_response, calculate_cached_cost,
    parse_trace_usage,
)


TERMINAL_ACTIONS = frozenset({"autosubmit", "exit", "finish", "finished", "submit"})
TERMINAL_TEXT = ("exit due to cost limit", "exited (autosubmitted)", "exited successfully")
STEP_MARKER = re.compile(r"=+ STEP (\d+) =+")


@dataclass(frozen=True)
class PrefixCostResult:
    cost: float | None
    status: str
    reason: str
    api_calls: int = 0
    trace_sha256: str | None = None


def actionable_indices(trajectory: list[dict]) -> tuple[int, ...]:
    indices = []
    for index, step in enumerate(trajectory):
        if not isinstance(step, dict):
            raise ValueError(f"trajectory step {index} is not an object")
        action = step.get("action")
        if not isinstance(action, str) or not action.strip():
            continue
        if action.strip().lower() in TERMINAL_ACTIONS:
            continue
        text = "\n".join(str(step.get(key) or "").lower() for key in
                         ("response", "output", "observation"))
        if any(marker in text for marker in TERMINAL_TEXT):
            continue
        indices.append(index)
    return tuple(indices)


def _billable_trajectory(trajectory: list[dict], api_calls: int) -> list[dict]:
    """Drop only a synthetic, unbilled terminal step appended by SWE-agent."""
    if len(trajectory) == api_calls:
        return trajectory
    if len(trajectory) == api_calls + 1:
        last = trajectory[-1]
        if (isinstance(last, dict) and not str(last.get("action") or "").strip()
                and any(marker in str(last.get("observation") or "").lower()
                        for marker in TERMINAL_TEXT)):
            return trajectory[:-1]
    raise ValueError("trajectory entries differ from reconciled API call count")


def _unavailable(reason: str) -> PrefixCostResult:
    return PrefixCostResult(None, "unavailable", reason)


def _trajectory(path: Path) -> tuple[list[dict], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("trajectory"), list):
        raise ValueError("trajectory is missing a step list")
    info = data.get("info")
    stats = info.get("model_stats") if isinstance(info, dict) else None
    if not isinstance(stats, dict):
        raise ValueError("trajectory model_stats are missing")
    expected = {"prompt_tokens": stats.get("tokens_sent"),
                "completion_tokens": stats.get("tokens_received"),
                "api_calls": stats.get("api_calls")}
    if any(type(value) is not int or value < 0 for value in expected.values()):
        raise ValueError("trajectory token and API counters are invalid")
    return data["trajectory"], expected


def _step_usages(trace: Path, identity: str, selected_start: int, endpoint: int) -> list[dict]:
    """Bind responses by API ordinal with monotone STEP marker provenance.

    A SWE-agent STEP can contain multiple model responses when a model turn
    does not produce an action. Its trajectory still records each response.
    Full API-call/trajectory count equality is checked by the caller.
    """
    usages: list[dict] = []
    marker_count = 0
    started = False
    expected_logger = {"swea-lm", f"swea-lm-{identity}"}
    agent_logger = {"swea-agent", f"swea-agent-{identity}"}
    with trace.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number < selected_start:
                continue
            match = _LOG_LINE.fullmatch(line.rstrip("\r\n"))
            if match is None:
                continue
            logger, message = match.groups()
            marker = STEP_MARKER.fullmatch(message.strip()) if logger in agent_logger else None
            if marker is not None:
                step_number = int(marker.group(1))
                # A selected retry's trajectory-start marker can precede the
                # previous attempt's final logged STEP. Bind only from its
                # own STEP 1, which is the first actionable API ordinal.
                if not started:
                    if step_number != 1:
                        continue
                    started = True
                if step_number <= marker_count:
                    raise ValueError("STEP markers are repeated or out of order")
                marker_count = step_number
            elif logger in expected_logger and message.startswith("Response:"):
                if not started:
                    continue
                usages.append(_parse_response(message[len("Response:"):].strip()))
                if len(usages) == endpoint:
                    return usages
    return usages


def prefix_cost_from_artifacts(
    traj_path: str | Path, trace_path: str | Path, k: int, full_cost: float,
    rates: Mapping[str, float],
) -> PrefixCostResult:
    """Return exact native-cache repricing through the Kth actionable step.

    `unavailable` includes a reason and no numeric estimate. The whole trace
    must reproduce the frozen five-model cost before prefix billing is trusted.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    traj_path, trace_path = Path(traj_path), Path(trace_path)
    if not traj_path.is_file():
        return _unavailable(f"missing trajectory: {traj_path}")
    if not trace_path.is_file():
        return _unavailable(f"missing trace: {trace_path}")
    try:
        trajectory, stats = _trajectory(traj_path)
        identity = traj_path.parent.name
        if traj_path.name != f"{identity}.traj":
            raise ValueError("trajectory path does not match artifact identity")
        usage = parse_trace_usage(trace_path, expected_instance_id=identity,
                                  expected_stats=stats)
        if usage["status"] != "parsed":
            raise ValueError("trace has no reconciled API responses")
        selection = usage["selection"]
        if "chain_recovery" in selection:
            raise ValueError("interleaved trace has no unique STEP-to-response binding")
        priced = calculate_cached_cost(usage, **rates)
        if not math.isclose(priced, full_cost, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"repriced full trace {priced} differs from frozen cost {full_cost}")
        trajectory = _billable_trajectory(trajectory, usage["api_calls"])
        actionable = actionable_indices(trajectory)
        if not actionable:
            raise ValueError("trajectory has no actionable step")
        endpoint = actionable[min(k, len(actionable)) - 1] + 1
        start = selection.get("run_selection", {}).get("selected_start_line", 1)
        step_usages = _step_usages(trace_path, identity, start, endpoint)
        if len(step_usages) != endpoint:
            raise ValueError("prefix STEP/API response count differs")
        total = sum(calculate_cached_cost(call, **rates) for call in step_usages)
        return PrefixCostResult(total, "exact", "", endpoint, usage["sha256"])
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, CachedBillingError) as exc:
        return _unavailable(str(exc))
