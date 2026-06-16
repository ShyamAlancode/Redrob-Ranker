"""Internal-consistency (honeypot) checks.

The dataset contains ~80 honeypot candidates with subtly impossible profiles,
forced to relevance tier 0. Ranking >10% honeypots in the top 100 is a
Stage 3 disqualification, so this module is defensive infrastructure, not an
optimisation.

We do not special-case known honeypots; we score *internal consistency* of
every profile, which is what the organisers say a good system should do
naturally. Checks are split into:

  hard flags -- arithmetic impossibilities a profile cannot honestly contain
  soft flags -- suspicious but individually explainable oddities
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .loading import career_span_years, months_between, parse_date


@dataclass
class IntegrityReport:
    hard_flags: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)

    @property
    def multiplier(self) -> float:
        if len(self.hard_flags) >= 2:
            return config.INTEGRITY_FATAL
        if len(self.hard_flags) == 1:
            return config.INTEGRITY_HARD
        return config.INTEGRITY_SOFT_DECAY ** len(self.soft_flags)

    @property
    def is_suspect(self) -> bool:
        return bool(self.hard_flags)


def check_integrity(candidate: dict) -> IntegrityReport:
    report = IntegrityReport()
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # -- 1. Claimed experience vs observable career span -----------------
    yoe = float(profile.get("years_of_experience") or 0.0)
    span = career_span_years(candidate)
    if history and yoe > span + config.HONEYPOT_YOE_SPAN_SLACK_YEARS:
        report.hard_flags.append(
            f"claims {yoe:.1f}y experience but career history spans only {span:.1f}y"
        )

    # -- 2. Stated role durations vs date arithmetic ----------------------
    for job in history:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or (
            config.REFERENCE_DATE if job.get("is_current") else None
        )
        stated = job.get("duration_months")
        if start and end and stated is not None:
            actual = months_between(start, end)
            if abs(actual - stated) > config.HONEYPOT_DURATION_MISMATCH_MONTHS:
                report.hard_flags.append(
                    f"role at {job.get('company', '?')} states {stated} months "
                    f"but dates span {actual:.0f}"
                )
        if start and end and end < start:
            report.hard_flags.append(
                f"role at {job.get('company', '?')} ends before it starts"
            )
        if job.get("is_current") and job.get("end_date"):
            report.soft_flags.append("current role has an end date")

    # -- 3. "Expert" skills never actually used ---------------------------
    hollow_experts = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(hollow_experts) >= config.HONEYPOT_EXPERT_ZERO_DURATION_MIN:
        report.hard_flags.append(
            f"{len(hollow_experts)} 'expert' skills with zero months of use"
        )
    elif hollow_experts:
        report.soft_flags.append(
            f"expert proficiency with no usage time: {', '.join(hollow_experts[:3])}"
        )

    # -- 4. Platform-signal oddities ---------------------------------------
    # Base-rate note (measured on the bundle's 50-candidate sample): salary
    # min>max occurs in ~26% of real profiles and skill-duration exceeding
    # total career in ~14% -- orders of magnitude too common to mark the ~80
    # honeypots (0.08% of pool), so they are treated as generator noise and
    # ignored. signup-after-last-active occurs in ~4%: suspicious enough to
    # note, far too common to hard-flag.
    signup = parse_date(signals.get("signup_date"))
    last_active = parse_date(signals.get("last_active_date"))
    if signup and last_active and last_active < signup:
        report.soft_flags.append("last active before signup date")

    listed = {s.get("name", "").lower() for s in skills}
    assessed = signals.get("skill_assessment_scores") or {}
    phantom = [name for name in assessed if name.lower() not in listed]
    if assessed and len(phantom) > len(assessed) / 2:
        report.soft_flags.append(
            f"assessments exist for unlisted skills: {', '.join(phantom[:3])}"
        )

    # -- 5. Micro-company long tenure (Type 3) -----------------------------
    for job in history:
        if (job.get("company_size") == "1-10" and 
            job.get("duration_months", 0) > 120):  # 10+ years at a micro company
            report.soft_flags.append("micro_company_long_tenure")

    # -- 6. Career duration inflation (Type 4) -----------------------------
    total_months = sum(j.get("duration_months", 0) for j in history)
    yoe_months = float(profile.get("years_of_experience") or 0.0) * 12
    if total_months > yoe_months + 24:  # more than 2 years gap
        report.soft_flags.append("career_duration_inflation")

    return report
