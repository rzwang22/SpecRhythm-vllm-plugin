"""Initial speculative budget allocation helpers."""

from __future__ import annotations

from vllm_specrhythm.scheduler.slo_state import SLOState


def fixed_budget(states: list[SLOState], budget_per_request: int) -> dict[str, int]:
    return {state.request_id: max(0, budget_per_request) for state in states}


def urgency_weighted_budget(
    states: list[SLOState],
    total_budget: int,
    max_per_request: int,
) -> dict[str, int]:
    if total_budget <= 0 or max_per_request <= 0:
        return {state.request_id: 0 for state in states}

    allocation = {state.request_id: 0 for state in states}
    ranked = sorted(
        states,
        key=lambda state: (
            state.urgency * state.expected_acceptance_benefit,
            state.urgency,
        ),
        reverse=True,
    )
    remaining = total_budget
    while remaining > 0:
        changed = False
        for state in ranked:
            if remaining <= 0:
                break
            if allocation[state.request_id] >= max_per_request:
                continue
            allocation[state.request_id] += 1
            remaining -= 1
            changed = True
        if not changed:
            break
    return allocation
