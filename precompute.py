#!/usr/bin/env python3
"""One-time precomputation script.
Loads candidates, normalizes skills and career history, computes static scores
(behavior and trust scores), and writes optimized candidate records to Parquet.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
import pandas as pd
import numpy as np

from ranker import config
from ranker.loading import iter_candidates, parse_date, months_between, career_span_years

def compute_contradiction_score(candidate: dict) -> float:
    """Check claimed AI/ML skills vs career history evidence."""
    claimed_ai_skills = 0
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "").lower()
        if any(ak in name for ak in config.AI_KEYWORDS):
            claimed_ai_skills += 1
            
    history = candidate.get("career_history", []) or []
    narrative = " ".join([j.get("description", "") or "" for j in history] + 
                         [j.get("title", "") or "" for j in history]).lower()
                         
    proven_ai_evidence = 0
    for kw in config.AI_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, narrative):
            proven_ai_evidence += 1

    # Check for marketing / non-tech roles
    non_tech_roles = 0
    for j in history:
        title = (j.get("title") or "").lower()
        if any(nt in title for nt in config.NON_TECH_TITLE_TERMS):
            non_tech_roles += 1
            
    total_roles = len(history) if history else 1
    non_tech_ratio = non_tech_roles / total_roles

    penalty = 0.0
    # Threshold check 1: Large mismatch
    if claimed_ai_skills >= 8 and proven_ai_evidence == 0:
        penalty = 25.0
    elif claimed_ai_skills >= 5 and proven_ai_evidence <= 1:
        penalty = 15.0

    # Threshold check 2: Marketing/non-tech career with claimed AI skills
    if claimed_ai_skills > 0 and non_tech_ratio > 0.5:
        diff = claimed_ai_skills - proven_ai_evidence
        if diff > 0:
            marketing_penalty = min(25.0, diff * 4.0 + non_tech_ratio * 10.0)
            penalty = max(penalty, marketing_penalty)
            
    return penalty



def precompute_behavior(candidate: dict) -> tuple[float, list[str], list[str]]:
    """Compute static behavior score, notes, and concerns."""
    signals = candidate.get("redrob_signals", {}) or {}
    notes = []
    concerns = []
    
    score = 80.0  # Base behavior score

    # Activity recency
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days = (config.REFERENCE_DATE - last_active).days
        if days > 180:
            score -= 50.0
            concerns.append(f"inactive on platform for ~{days} days")
        elif days > 90:
            score -= 30.0
            concerns.append(f"inactive on platform for ~{days} days")
        elif days > 45:
            score -= 15.0
        elif days > 14:
            score -= 5.0
        else:
            notes.append("active on the platform this fortnight")

    # Response rate
    rate = signals.get("recruiter_response_rate")
    if rate is not None:
        if rate < 0.1:
            score -= 40.0
            concerns.append(f"{rate:.0%} recruiter response rate")
        elif rate < 0.3:
            score -= 25.0
            concerns.append(f"{rate:.0%} recruiter response rate")
        elif rate < 0.6:
            score -= 10.0
        else:
            notes.append(f"{rate:.0%} recruiter response rate")

    # Open to work
    if signals.get("open_to_work_flag"):
        score += 5.0
        notes.append("open to work")
    else:
        score -= 10.0

    # Interview completion
    icr = signals.get("interview_completion_rate")
    if icr is not None and icr < 0.5:
        score -= 15.0
        concerns.append(f"completes only {icr:.0%} of scheduled interviews")

    # GitHub
    gh = signals.get("github_activity_score")
    if gh is not None:
        if gh >= 50:
            score += 3.0
            notes.append(f"active public GitHub (score {gh:.0f})")
        elif gh == -1:
            score -= 3.0

    # Verification
    if signals.get("verified_email") and signals.get("verified_phone"):
        score += 2.0

    # Saves
    saves = signals.get("saved_by_recruiters_30d") or 0
    if saves > 0:
        score += min(saves, 5) * 1.0
        notes.append(f"saved by {saves} recruiter(s) in the last 30 days")

    # Views
    views = signals.get("profile_views_received_30d") or 0
    if views >= config.PROFILE_VIEWS_THRESHOLD:
        score += 1.0

    # Applications
    apps = signals.get("applications_submitted_30d") or 0
    if apps > 0:
        score += 2.0
        notes.append("actively applying to roles")

    # Cap behavior score between 0.0 and 100.0
    behavior_score = float(max(0.0, min(100.0, score)))
    return behavior_score, notes, concerns

def precompute_trust(candidate: dict, concerns_list: list[str]) -> float:
    """Compute static trust score and collect structural concerns."""
    signals = candidate.get("redrob_signals", {}) or {}
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    
    trust_score = 100.0

    # AI/ML keywords for contradiction and consulting checks
    ai_keywords = {"llm", "nlp", "gan", "gans", "pytorch", "tensorflow", "embedding", "embeddings", 
                   "vector", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "retrieval", 
                   "ranking", "recommendation", "rag", "lora", "qlora", "peft", "transformers"}

    # 1. Contradiction penalty
    contradiction_penalty = compute_contradiction_score(candidate)
    if contradiction_penalty > 0:
        trust_score -= contradiction_penalty
        concerns_list.append(f"AI skill list doesn't match a '{profile.get('current_title')}' career")

    # 2. Suspicious AI skill stuffing
    hollow_skills = sum(1 for s in skills if s.get("duration_months", 0) == 0 and s.get("proficiency") in ("expert", "advanced"))
    if hollow_skills >= config.HONEYPOT_EXPERT_ZERO_DURATION_MIN:
        trust_score -= min(20.0, hollow_skills * 5.0)
        concerns_list.append(f"{hollow_skills} 'expert' skills with zero months of use")

    # 3. Incomplete profiles
    profile_comp = signals.get("profile_completeness_score", 100.0)
    if profile_comp < 70.0:
        trust_score -= (70.0 - profile_comp) * 0.3

    # 4. Unverifiable contact
    if not signals.get("verified_email") or not signals.get("verified_phone"):
        trust_score -= 10.0

    # 5. Core structural check penalties (as soft penalties in trust_score)
    title = (profile.get("current_title") or "").lower()
    narrative = " ".join(
        [(profile.get("summary") or "")] + [(j.get("description") or "") for j in history] + [(j.get("title") or "") for j in history]
    ).lower()

    # Consulting background check
    has_consulting = False
    if history:
        services = [
            j for j in history
            if any(ind in (j.get("industry") or "").lower() for ind in config.CONSULTING_INDUSTRIES)
            or any(firm in (j.get("company") or "").lower() for firm in config.CONSULTING_FIRMS)
        ]
        if len(services) > 0:
            has_consulting = True

    # Check for AI/ML evidence in skills or career history narrative
    has_ai_skill = False
    for s in skills:
        name = (s.get("name") or "").lower()
        if any(ak in name for ak in ai_keywords):
            has_ai_skill = True
            break
            
    has_ai_narrative = any(ak in narrative for ak in ai_keywords) or \
                       any(mt in narrative for mt in config.ML_EVIDENCE_TERMS) or \
                       any(rt in narrative for rt in config.RETRIEVAL_EVIDENCE_TERMS)
                       
    has_ai_evidence = has_ai_skill or has_ai_narrative

    # Penalize ONLY if consulting background AND no AI/ML evidence is found
    if has_consulting and not has_ai_evidence:
        trust_score -= config.PENALTY_CONSULTING_ONLY
        concerns_list.append("consulting background with no visible AI/ML production evidence")

    # Research only
    if history and all(
        any(rt in (j.get("title") or "").lower() for rt in config.RESEARCH_TITLE_TERMS)
        or (j.get("industry") or "").lower() in config.RESEARCH_INDUSTRIES
        for j in history
    ) and sum(1 for p in config.PRODUCTION_EVIDENCE_TERMS if p in narrative) == 0:
        trust_score -= config.PENALTY_RESEARCH_ONLY
        concerns_list.append("pure research background with no production deployment signal")

    # Title chaser
    yoe = float(profile.get("years_of_experience") or 0.0)
    if len(history) >= config.TITLE_CHASER_MIN_ROLES and yoe >= 4:
        tenures = [j.get("duration_months", 0) or 0 for j in history]
        if tenures and sum(tenures) / len(tenures) < config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
            trust_score -= config.PENALTY_TITLE_CHASER
            concerns_list.append("frequent short stints (avg tenure < 20 months)")

    # CV only
    cv_hits = sum(1 for cv in config.CV_SPEECH_ROBOTICS_TERMS if cv in narrative)
    nlp_hits = sum(1 for nlp in config.NLP_IR_TERMS if nlp in narrative)
    if cv_hits >= 3 and nlp_hits == 0:
        trust_score -= config.PENALTY_CV_ONLY
        concerns_list.append("primary expertise in CV/speech/robotics with no NLP/IR exposure")

    # Stale hands on
    current = next((j for j in history if j.get("is_current")), None)
    if current and any(lead in (current.get("title") or "").lower() for lead in config.LEADERSHIP_TITLE_TERMS):
        months = current.get("duration_months", 0) or 0
        desc = (current.get("description") or "").lower()
        if months >= config.STALE_HANDS_ON_MONTHS and not any(verb in desc for verb in config.HANDS_ON_VERBS):
            trust_score -= config.PENALTY_STALE_HANDS_ON
            concerns_list.append("18+ months in a non-coding leadership role")

    return float(max(0.0, min(100.0, trust_score)))

def process_candidate(candidate: dict) -> dict:
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    
    candidate_id = candidate.get("candidate_id", "")
    yoe = float(profile.get("years_of_experience") or 0.0)
    
    # Skills normalization
    skills = []
    for s in candidate.get("skills", []) or []:
        name = (s.get("name") or "").lower().strip()
        if name:
            skills.append({
                "name": name,
                "proficiency": (s.get("proficiency") or "intermediate").lower(),
                "duration_months": int(s.get("duration_months") or 0),
                "endorsements": int(s.get("endorsements") or 0)
            })
            
    # Career keywords
    history = candidate.get("career_history", []) or []
    words = set()
    for job in history:
        title = (job.get("title") or "").lower()
        desc = (job.get("description") or "").lower()
        for w in re.findall(r"\b[a-z0-9_-]+\b", title + " " + desc):
            words.add(w)
            
    # Education parsing
    education = []
    for edu in candidate.get("education", []) or []:
        education.append({
            "institution": edu.get("institution") or "",
            "degree": edu.get("degree") or "",
            "field_of_study": edu.get("field_of_study") or "",
            "tier": edu.get("tier") or "unknown"
        })

    # Precompute career evidence keywords (saves runtime rank time)
    narrative_evidence = " ".join([j.get("description", "") or "" for j in history] + 
                                  [j.get("title", "") or "" for j in history]).lower()
    matched_evidence = []
    evidence_candidates = [
        "ranking pipelines", "ml infrastructure", "recommendation engine", 
        "semantic search", "retrieval systems", "production models", 
        "vector search", "search infrastructure", "information retrieval",
        "hybrid search", "eval frameworks", "ab testing", "recommendation systems",
        "retrieval", "ranking", "search relevance", "embeddings"
    ]
    for term in evidence_candidates:
        if term in narrative_evidence:
            matched_evidence.append(term)
            if len(matched_evidence) >= 2:
                break
    if not matched_evidence:
        matched_evidence = ["software engineering"]

    # Compute static scores & concerns
    behavior_score, behavior_notes, behavior_concerns = precompute_behavior(candidate)
    
    structural_concerns = []
    trust_score = precompute_trust(candidate, structural_concerns)
    
    # Notice and work mode concerns (based on JD logic)
    notice = int(signals.get("notice_period_days") or 90)
    if notice > 60:
        structural_concerns.append(f"{notice}-day notice period")
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate"))
    if country and country != "india":
        structural_concerns.append(f"based in {profile.get('location')}, {profile.get('country')}")
    elif not any(pref in location for pref in config.LOCATION_PREFERRED) and not any(welcome in location for welcome in config.LOCATION_WELCOME) and not relocate:
        structural_concerns.append("outside Noida/Pune/welcome cities and not willing to relocate")

    # Combine concerns
    all_concerns = structural_concerns + behavior_concerns
    
    # Lightweight metadata for reasoning
    meta = {
        "current_title": profile.get("current_title") or "",
        "current_company": profile.get("current_company") or "",
        "location": profile.get("location") or "",
        "country": profile.get("country") or "",
        "headline": profile.get("headline") or "",
        "summary": profile.get("summary") or "",
        "notice_period_days": notice,
        "willing_to_relocate": relocate,
        "preferred_work_mode": signals.get("preferred_work_mode", "flexible"),
        "last_active_date": signals.get("last_active_date"),
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "github_activity_score": signals.get("github_activity_score"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "verified_email": signals.get("verified_email"),
        "verified_phone": signals.get("verified_phone"),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d"),
        "profile_views_received_30d": signals.get("profile_views_received_30d"),
        "applications_submitted_30d": signals.get("applications_submitted_30d"),
        "concerns": all_concerns,
        "notes": behavior_notes,
        "education": education,
        "career_evidence": matched_evidence
    }
    
    return {
        "candidate_id": candidate_id,
        "years_of_experience": yoe,
        "normalized_skill_set": json.dumps(skills),
        "career_keyword_set": json.dumps(list(words)),
        "static_behavior_score": behavior_score,
        "trust_score": trust_score,
        "metadata": json.dumps(meta)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="artifacts/processed_candidates.parquet", help="Path to write Parquet output")
    args = parser.parse_args()

    started = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Precomputing static scores for candidates from {args.candidates}...")
    records = []
    for candidate in iter_candidates(args.candidates):
        records.append(process_candidate(candidate))

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} candidates to {out_path} in {time.time() - started:.1f}s")

if __name__ == "__main__":
    main()
