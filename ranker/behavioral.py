"""Behavioral availability multiplier.

The JD's hackathon note is explicit: "a perfect-on-paper candidate who hasn't
logged in for 6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available. Down-weight them appropriately."

This is implemented as a *multiplier* rather than an additive component on
purpose: strong behavioral signals should never compensate for a weak skills
fit (an enthusiastic accountant is still an accountant), but weak behavioral
signals should drag down even a perfect fit. Multiplication gives exactly
that asymmetry; addition does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .loading import parse_date


@dataclass
class BehavioralResult:
    multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    days_inactive: int | None = None
    response_rate: float | None = None


def behavioral_multiplier(candidate: dict) -> BehavioralResult:
    result = BehavioralResult()
    signals = candidate.get("redrob_signals", {}) or {}
    m = 1.0

    # -- Recency of activity ----------------------------------------------
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days = (config.REFERENCE_DATE - last_active).days
        result.days_inactive = days
        step = config.ACTIVITY_DECAY_STALE
        for max_days, value in config.ACTIVITY_DECAY:
            if days <= max_days:
                step = value
                break
        m *= step
        if days > 90:
            result.concerns.append(f"inactive on platform for ~{days} days")
        elif days <= 14:
            result.notes.append("active on the platform this fortnight")

    # -- Responsiveness -----------------------------------------------------
    rate = signals.get("recruiter_response_rate")
    if rate is not None:
        result.response_rate = rate
        step = config.RESPONSE_RATE_FLOOR
        for min_rate, value in config.RESPONSE_RATE_STEPS:
            if rate >= min_rate:
                step = value
                break
        m *= step
        if rate < 0.2:
            result.concerns.append(f"{rate:.0%} recruiter response rate")
        elif rate >= 0.6:
            result.notes.append(f"{rate:.0%} recruiter response rate")

    # -- Stated availability ------------------------------------------------
    if signals.get("open_to_work_flag"):
        m *= 1.05
        result.notes.append("open to work")
    else:
        m *= 0.90

    # -- Process reliability --------------------------------------------------
    icr = signals.get("interview_completion_rate")
    if icr is not None and icr < 0.5:
        m *= 0.85
        result.concerns.append(f"completes only {icr:.0%} of scheduled interviews")

    # -- External validation ("we need to see how you think") ----------------
    # JD disqualifier proxy: 5+ years of closed-source work with no external
    # validation. github_activity_score is the only observable signal for it
    # (-1 = no GitHub linked), so the adjustment is deliberately small.
    gh = signals.get("github_activity_score")
    if gh is not None:
        if gh >= 50:
            m *= 1.03
            result.notes.append(f"active public GitHub (score {gh:.0f})")
        elif gh == -1:
            m *= 0.97

    # -- Identity verification (small, but cheap trust signal) ---------------
    if signals.get("verified_email") and signals.get("verified_phone"):
        m *= 1.02

    result.multiplier = max(config.BEHAVIORAL_FLOOR, min(config.BEHAVIORAL_CEILING, m))
    return result
