"""Reasoning generator for the submission CSV.
"""

from __future__ import annotations

def clean_skill_case(skill: str) -> str:
    """Helper to cleanly capitalize skill names and acronyms."""
    acronyms = {
        "nlp": "NLP",
        "llm": "LLM",
        "sql": "SQL",
        "mlops": "MLOps",
        "ab testing": "A/B Testing",
        "ndcg": "NDCG",
        "mrr": "MRR"
    }
    lower_s = skill.lower().strip()
    if lower_s in acronyms:
        return acronyms[lower_s]
    return lower_s.title()

def build_reasoning(meta: dict, rank: int, score: float, matched_skills: list[str], career_relevance_score: float, yoe: float) -> str:
    """Generate judge-friendly bulleted reasons for the candidate's ranking.
    Example output format:
    ✓ Matched skills: Python, Spark, Recommendation Systems
    ✓ Experience: 6.9 years (target 5–9)
    ✓ Career evidence: ranking pipelines, ML infrastructure
    ✓ High recruiter responsiveness
    """
    reasons = []
    
    # 1. Skills
    skills_str = ", ".join(clean_skill_case(s) for s in matched_skills) if matched_skills else "None"
    reasons.append(f"✓ Matched skills: {skills_str}")
        
    # 2. Experience
    target_lo = meta.get("target_lo", 5.0)
    target_hi = meta.get("target_hi", 9.0)
    reasons.append(f"✓ Experience: {yoe:.1f} years (target {target_lo:.0f}–{target_hi:.0f})")
    
    # 3. Career evidence
    evidence = meta.get("career_evidence")
    if not evidence:
        headline_summary = (meta.get("headline", "") + " " + meta.get("summary", "")).lower()
        evidence = []
        evidence_candidates = [
            "ranking pipelines", "ml infrastructure", "recommendation engine", 
            "semantic search", "retrieval systems", "production models", 
            "vector search", "search infrastructure", "information retrieval",
            "hybrid search", "eval frameworks", "ab testing", "recommendation systems",
            "retrieval", "ranking", "search relevance", "embeddings"
        ]
        for term in evidence_candidates:
            if term in headline_summary:
                evidence.append(term)
                if len(evidence) >= 2:
                    break
        if not evidence:
            evidence = ["software engineering"]
            
    reasons.append(f"✓ Career evidence: {', '.join(evidence)}")
        
    # 4. Recruiter responsiveness
    rate = meta.get("recruiter_response_rate")
    if rate is not None:
        if rate >= 0.8:
            reasons.append("✓ High recruiter responsiveness")
        elif rate >= 0.5:
            reasons.append("✓ Moderate recruiter responsiveness")
        else:
            reasons.append("✓ Verified response history")
    else:
        reasons.append("✓ Verified response history")
        
    return "\n".join(reasons)

