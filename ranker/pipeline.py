"""Pipeline orchestration: combine rule-based metrics into final score,
select top-K using heapq, and write submission CSV.
"""

from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from . import config
from .structural import (
    score_career_relevance,
    score_skills_match,
    score_experience_match,
    score_education
)
from .reasoning import build_reasoning

@dataclass
class ScoredCandidate:
    candidate_id: str
    final: float
    career_relevance: float
    skills_match: float
    experience_match: float
    behavior_score: float
    trust_score: float
    matched_skills: list[str]
    yoe: float
    meta: dict
    education_match: float = 0.0

    def sort_key(self) -> tuple:
        # Higher score first, ties broken by candidate_id ascending.
        return (-self.final, self.candidate_id)

def score_candidate(
    row: dict,
    jd_required_skills: list[str],
    jd_preferred_skills: list[str],
    jd_min_experience: float,
    jd_role_keywords: list[str],
    weights: dict | None = None,
    jd_max_experience: float | None = None
) -> ScoredCandidate:
    candidate_id = row["candidate_id"]
    yoe = float(row["years_of_experience"])
    
    # Parse json fields
    cand_skills = json.loads(row["normalized_skill_set"])
    cand_skills_names = [s["name"] for s in cand_skills]
    
    cand_keywords = set(json.loads(row["career_keyword_set"]))
    
    behavior_score = float(row["static_behavior_score"])
    trust_score = float(row["trust_score"])
    
    # Calculate JD-dependent scores
    career_relevance = score_career_relevance(cand_keywords, jd_role_keywords)
    skills_match = score_skills_match(cand_skills_names, jd_required_skills, jd_preferred_skills)
    experience_match = score_experience_match(yoe, jd_min_experience)
    
    meta = json.loads(row["metadata"])
    education_match = score_education(meta.get("education", []))
    
    # Get weights
    if weights is None:
        weights = {
            "career_relevance": config.W_CAREER_RELEVANCE,
            "skills_match": config.W_SKILLS_MATCH,
            "experience_match": config.W_EXPERIENCE_MATCH,
            "behavior_score": config.W_BEHAVIOR_SCORE,
            "trust_score": config.W_TRUST_SCORE,
            "education_match": 0.0
        }
    
    # Additive weighted score
    final_score = (
        weights["career_relevance"] * career_relevance +
        weights["skills_match"] * skills_match +
        weights["experience_match"] * experience_match +
        weights["behavior_score"] * behavior_score +
        weights["trust_score"] * trust_score +
        weights.get("education_match", 0.0) * education_match
    )
    
    # Identify matched skills for reasoning
    matched_skills = list(set(cand_skills_names).intersection(jd_required_skills + jd_preferred_skills))
    
    # Store experience targets in meta for reasoning
    meta["target_lo"] = jd_min_experience
    meta["target_hi"] = jd_max_experience if jd_max_experience is not None else jd_min_experience + 4.0
    
    return ScoredCandidate(
        candidate_id=candidate_id,
        final=final_score,
        career_relevance=career_relevance,
        skills_match=skills_match,
        experience_match=experience_match,
        behavior_score=behavior_score,
        trust_score=trust_score,
        matched_skills=matched_skills,
        yoe=yoe,
        meta=meta,
        education_match=education_match
    )

class _InvertedStr(str):
    """String with reversed comparison, for min-heap tiebreak ordering."""
    def __lt__(self, other):  # type: ignore[override]
        return str.__gt__(self, other)
    def __gt__(self, other):  # type: ignore[override]
        return str.__lt__(self, other)

def select_top(
    candidates_df: pd.DataFrame,
    jd_info: dict,
    top_k: int = 100
) -> list[ScoredCandidate]:
    """Score candidates and maintain top K using a min-heap (O(N log K))."""
    heap: list[tuple[float, _InvertedStr, ScoredCandidate]] = []
    
    req_skills = jd_info["required_skills"]
    pref_skills = jd_info["preferred_skills"]
    min_exp = jd_info["min_experience"]
    max_exp = jd_info.get("max_experience")
    role_kws = jd_info["role_keywords"]
    
    # Load dynamic weights based on the JD text
    weights = config.get_dynamic_weights(jd_info.get("jd_text", ""))
    
    # Iterate through candidates
    for row in candidates_df.to_dict('records'):
        scored = score_candidate(row, req_skills, pref_skills, min_exp, role_kws, weights=weights, jd_max_experience=max_exp)
        entry = (scored.final, _InvertedStr(scored.candidate_id), scored)
        
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
            
    # Sort results correctly
    finalists = sorted((e[2] for e in heap), key=ScoredCandidate.sort_key)
    return finalists

def write_submission(ranked: list[ScoredCandidate], out_path: str | Path) -> None:
    """Write the spec-compliant CSV: header, exactly len(ranked) rows,
    non-increasing scores, UTF-8.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decorated = sorted(
        ((min(round(sc.final, 4), 100.0), sc) for sc in ranked),
        key=lambda pair: (-pair[0], pair[1].candidate_id),
    )

    rows = []
    for i, (score, sc) in enumerate(decorated, start=1):
        rows.append({
            "candidate_id": sc.candidate_id,
            "rank": i,
            "score": f"{score:.4f}",
            "reasoning": build_reasoning(
                sc.meta,
                i,
                score,
                sc.matched_skills,
                sc.career_relevance,
                sc.yoe
            ),
        })

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
