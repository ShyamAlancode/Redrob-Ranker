"""Tests for the refactored offline rule-based Redrob-Ranker.
Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import pytest
import pandas as pd

from precompute import (
    process_candidate,
    precompute_behavior,
    precompute_trust,
    compute_contradiction_score
)
from ranker.structural import (
    parse_job_description,
    score_career_relevance,
    score_skills_match,
    score_experience_match
)
from ranker.pipeline import score_candidate, select_top, write_submission
from ranker.reasoning import build_reasoning

def make_candidate(cid: str = "CAND_0000001", **overrides) -> dict:
    """Fixture to build a clean baseline candidate."""
    base = {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test Person",
            "headline": "ML Engineer | Search & Ranking",
            "summary": "7 years building embeddings-based retrieval and ranking systems "
                       "shipped to production at product companies.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Machine Learning Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "ProductCo",
                "title": "Machine Learning Engineer",
                "start_date": "2022-06-01",
                "end_date": None,
                "duration_months": 48,
                "is_current": True,
                "industry": "Software",
                "company_size": "201-500",
                "description": "Built and deployed an embeddings-based semantic search and "
                               "ranking system serving real users in production; owned A/B "
                               "evaluation with NDCG.",
            },
            {
                "company": "EarlierCo",
                "title": "Software Engineer",
                "start_date": "2019-06-01",
                "end_date": "2022-05-01",
                "duration_months": 35,
                "is_current": False,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Implemented a recommendation engine and search relevance "
                               "improvements for an e-commerce platform.",
            },
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 80},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 25, "duration_months": 48},
            {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 18, "duration_months": 36},
            {"name": "PyTorch", "proficiency": "advanced", "endorsements": 20, "duration_months": 40},
        ],
        "redrob_signals": {
            "profile_completeness_score": 90.0,
            "signup_date": "2025-01-15",
            "last_active_date": "2026-05-28",
            "open_to_work_flag": True,
            "profile_views_received_30d": 30,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 88.0},
            "connection_count": 400,
            "endorsements_received": 60,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60.0,
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def test_jd_parser():
    """Verify dynamic Job Description parsing extracts metrics successfully."""
    jd_content = (
        "Experience Required: 6–10 years\n"
        "Things you absolutely need\n"
        "Strong Python. Production experience with embeddings and vector databases.\n"
        "Things we'd like you to have\n"
        "LLM fine-tuning experience (LoRA, QLoRA, PEFT) and MLOps.\n"
        "Mandate: own the retrieval and ranking layers."
    )
    parsed = parse_job_description(jd_content)
    assert parsed["min_experience"] == 6
    assert "python" in parsed["required_skills"]
    assert "mlops" in parsed["preferred_skills"]
    assert "retrieval" in parsed["role_keywords"]
    assert "ranking" in parsed["role_keywords"]

def test_contradiction_score():
    """Verify contradiction detector triggers on non-tech titles with AI skills."""
    # A clean ML candidate should have 0 contradiction score
    clean = make_candidate()
    assert compute_contradiction_score(clean) == 0.0

    # A candidate claiming AI skills but working in Marketing should get penalized
    stuffer = make_candidate()
    stuffer["profile"]["current_title"] = "Marketing Manager"
    for job in stuffer["career_history"]:
        job["title"] = "Marketing Manager"
        job["description"] = "Managed offline marketing campaigns and sales brochures."
    
    # AI skills claimed (must be >= 5 to trigger penalty)
    stuffer["skills"] = [
        {"name": "LLM", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
        {"name": "Deep Learning", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
        {"name": "NLP", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
        {"name": "Machine Learning", "proficiency": "expert", "endorsements": 10, "duration_months": 24},
        {"name": "Retrieval", "proficiency": "expert", "endorsements": 10, "duration_months": 24}
    ]
    
    penalty = compute_contradiction_score(stuffer)
    assert penalty > 0.0


def test_precompute_behavior_score():
    """Verify static behavior score ranges and updates correctly."""
    c = make_candidate()
    score, notes, concerns = precompute_behavior(c)
    assert 0.0 <= score <= 100.0
    assert len(notes) > 0
    assert len(concerns) == 0

def test_precompute_trust_score():
    """Verify trust score computation and soft additive penalties."""
    c = make_candidate()
    concerns = []
    score = precompute_trust(c, concerns)
    # Clean candidate should have high trust score
    assert score >= 90.0

    # Contradictory / unverified candidate should have lower trust
    untrusty = make_candidate()
    untrusty["redrob_signals"]["verified_email"] = False
    untrusty["redrob_signals"]["verified_phone"] = False
    untrusty_score = precompute_trust(untrusty, [])
    assert untrusty_score < score

def test_scoring_weights():
    """Verify that score_candidate applies correct weights from config."""
    c = make_candidate()
    proc = process_candidate(c)
    
    jd_info = {
        "required_skills": ["python", "embeddings"],
        "preferred_skills": ["mlops"],
        "min_experience": 5,
        "role_keywords": ["ranking", "recommendation"]
    }
    
    scored = score_candidate(
        proc,
        jd_info["required_skills"],
        jd_info["preferred_skills"],
        jd_info["min_experience"],
        jd_info["role_keywords"]
    )
    
    assert 0.0 <= scored.final <= 100.0
    assert scored.career_relevance > 0.0
    assert scored.skills_match > 0.0

def test_select_top_heap_ranking():
    """Verify select_top returns ranked finalists with proper tie breaking."""
    # Build multiple mock candidates
    c1 = make_candidate("CAND_0000001")
    c2 = make_candidate("CAND_0000002") # Identical but lexicographically larger ID
    
    proc1 = process_candidate(c1)
    proc2 = process_candidate(c2)
    
    df = pd.DataFrame([proc1, proc2])
    
    jd_info = {
        "required_skills": ["python"],
        "preferred_skills": [],
        "min_experience": 5,
        "role_keywords": ["ranking"]
    }
    
    top = select_top(df, jd_info, top_k=2)
    assert len(top) == 2
    # CAND_0000001 must rank before CAND_0000002 on tie break
    assert top[0].candidate_id == "CAND_0000001"
    assert top[1].candidate_id == "CAND_0000002"

def test_reasoning_format():
    """Verify reasoning format uses the judge-friendly checkmark checklist."""
    meta = {
        "notes": ["active on the platform this fortnight"],
        "open_to_work_flag": True,
        "target_lo": 5.0,
        "target_hi": 9.0,
        "career_evidence": ["ranking pipelines", "ml infrastructure"],
        "recruiter_response_rate": 0.8
    }
    reasoning = build_reasoning(
        meta=meta,
        rank=1,
        score=95.0,
        matched_skills=["python", "recommendation systems"],
        career_relevance_score=85.0,
        yoe=7.0
    )
    lines = reasoning.splitlines()
    assert len(lines) == 4
    assert all(line.startswith("✓ ") for line in lines)
    assert "Python" in lines[0]
    assert "Recommendation Systems" in lines[0]
    assert "Experience: 7.0 years (target 5–9)" in lines[1]
    assert "Career evidence: ranking pipelines, ml infrastructure" in lines[2]
    assert "High recruiter responsiveness" in lines[3]


def test_write_submission_csv(tmp_path: Path):
    """Verify CSV submission file has compliant column structure and order."""
    c1 = make_candidate("CAND_0000001")
    proc = process_candidate(c1)
    
    jd_info = {
        "required_skills": ["python"],
        "preferred_skills": [],
        "min_experience": 5,
        "role_keywords": ["ranking"]
    }
    
    df = pd.DataFrame([proc])
    ranked = select_top(df, jd_info, top_k=1)
    
    out_file = tmp_path / "submission.csv"
    write_submission(ranked, out_file)
    
    assert out_file.exists()
    with open(out_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 1
    assert list(rows[0].keys()) == ["candidate_id", "rank", "score", "reasoning"]
    assert rows[0]["candidate_id"] == "CAND_0000001"
    assert rows[0]["rank"] == "1"
    assert "✓" in rows[0]["reasoning"]
