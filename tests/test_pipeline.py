"""Tests for the redrob-ranker scoring pipeline.

Run with: python -m pytest tests/ -v

These guard the failure modes that actually disqualify submissions:
honeypots in the top 100 (Stage 3), tie-break/format violations (Stage 1),
and keyword stuffers outranking real engineers (the scored trap).
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

from ranker.behavioral import behavioral_multiplier
from ranker.honeypot import check_integrity
from ranker.pipeline import score_candidate, select_top, write_submission
from ranker.structural import structural_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_candidate(cid: str = "CAND_0000001", **overrides) -> dict:
    """A clean, strong-fit candidate; tests mutate from this baseline."""
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


# ---------------------------------------------------------------------------
# Honeypot / integrity
# ---------------------------------------------------------------------------

def test_clean_candidate_has_no_hard_flags():
    assert check_integrity(make_candidate()).hard_flags == []


def test_impossible_experience_span_is_hard_flagged():
    # Claims 8 years but earliest start date is mid-2024.
    c = make_candidate(profile={"years_of_experience": 8.0})
    c["career_history"] = [c["career_history"][0]]
    c["career_history"][0]["start_date"] = "2024-06-01"
    c["career_history"][0]["duration_months"] = 24
    report = check_integrity(c)
    assert any("spans only" in f for f in report.hard_flags)


def test_hollow_expert_skills_are_hard_flagged():
    c = make_candidate()
    c["skills"] = [
        {"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for s in ("RAG", "Pinecone", "LLM", "Embeddings")
    ]
    report = check_integrity(c)
    assert any("zero months" in f for f in report.hard_flags)


def test_duration_date_mismatch_is_hard_flagged():
    c = make_candidate()
    c["career_history"][1]["duration_months"] = 90  # dates span ~35 months
    report = check_integrity(c)
    assert any("dates span" in f for f in report.hard_flags)


def test_noisy_signals_are_soft_not_hard():
    """Salary inversion appears in ~26% of real profiles and signup-after-
    activity in ~4% (measured on the bundle sample) -- they are generator
    noise, not honeypot markers, and must never hard-flag a candidate."""
    c = make_candidate(redrob_signals={
        "expected_salary_range_inr_lpa": {"min": 50, "max": 20},
        "signup_date": "2026-05-01",
        "last_active_date": "2026-01-01",
    })
    report = check_integrity(c)
    assert report.hard_flags == []
    assert report.multiplier > 0.8  # mild soft penalty at most


def test_two_hard_flags_effectively_exclude():
    c = make_candidate(profile={"years_of_experience": 9.0})
    c["career_history"] = [c["career_history"][0]]
    c["career_history"][0]["start_date"] = "2024-09-01"
    c["career_history"][0]["duration_months"] = 90  # also mismatches dates
    report = check_integrity(c)
    assert len(report.hard_flags) >= 2
    assert report.multiplier <= 0.05  # effectively excluded


# ---------------------------------------------------------------------------
# Antigravity audit: 4 bug fixes
# ---------------------------------------------------------------------------

def test_concern_always_stated_when_present(tmp_path: Path):
    """Concerns must appear in the reasoning for EVERY rank combination,
    not just when modulo happens to select a concern-pool option."""
    candidates = [make_candidate(cid=f"CAND_{i:07d}") for i in range(1, 101)]
    # Give all candidates a concern: long notice period.
    for c in candidates:
        c["redrob_signals"]["notice_period_days"] = 120
    lookup = {c["candidate_id"]: 0.9 - i * 0.008 for i, c in enumerate(candidates)}.get
    ranked = select_top(iter(candidates), lookup, top_k=100)
    out = tmp_path / "team_test.csv"
    write_submission(ranked, out)
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Every reasoning row must acknowledge the notice period concern.
    violations = [r for r in rows if "notice" not in r["reasoning"].lower()
                  and "No material gaps" in r["reasoning"]]
    assert violations == [], (
        f"Rows claimed 'No material gaps' despite 120-day notice: "
        f"{[v['rank'] for v in violations]}"
    )


def test_cv_only_not_triggered_by_context_substring():
    """'context' must not bypass the CV-only penalty via substring match for 'text'."""
    c = make_candidate(cid="CAND_0000050")
    # Wipe the retrieval-heavy career history so we're testing CV-only detection cleanly.
    c["profile"]["summary"] = (
        "Computer vision engineer specializing in image segmentation and object "
        "detection. Used CNN and OpenCV extensively in the context of autonomous "
        "driving and robotics pipelines."
    )
    c["profile"]["headline"] = "Computer Vision Engineer | Robotics"
    c["career_history"] = [{
        "company": "RoboticsCo", "title": "CV Engineer",
        "start_date": "2019-01-01", "end_date": None,
        "duration_months": 72, "is_current": True, "industry": "Robotics",
        "company_size": "51-200",
        "description": (
            "Developed object detection models using YOLO and EfficientDet for "
            "autonomous navigation. Used OpenCV for image preprocessing pipelines "
            "in the context of real-time inference. No NLP or text work."
        ),
    }]
    c["skills"] = [
        {"name": "Computer Vision", "proficiency": "expert", "endorsements": 30, "duration_months": 60},
        {"name": "OpenCV", "proficiency": "expert", "endorsements": 20, "duration_months": 48},
        {"name": "Image Classification", "proficiency": "expert", "endorsements": 25, "duration_months": 48},
        {"name": "Object Detection", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
    ]
    result = structural_score(c)
    # Old code: "context" matched "text" via plain substring → nlp_hits=1 → no penalty
    # New code: \btext\b does NOT match inside "context" → nlp_hits=0 → penalty fires
    assert "cv_only" in result.penalties, (
        f"cv_only not detected. nlp terms matched: "
        f"{[t for t in ('nlp','text','search','ranking','retrieval','recommendation','embedding') if t in result.concerns]}"
    )


def test_consulting_industry_substring_variants():
    """'IT Services & Consulting' and 'IT Consulting' must be caught."""
    for industry in ("IT Services & Consulting", "IT Consulting", "IT Services"):
        c = make_candidate(cid="CAND_0000060")
        for job in c["career_history"]:
            job["industry"] = industry
        result = structural_score(c)
        assert "consulting_only" in result.penalties, (
            f"consulting_only not fired for industry='{industry}'"
        )


def test_langchain_only_is_penalized():
    """Candidate whose ML history is dominated by LangChain/OpenAI wrappers
    with no pre-LLM production narrative must receive the langchain_only penalty."""
    c = make_candidate(cid="CAND_0000070")
    # Override the ML-engineer base so only LangChain evidence remains.
    c["profile"]["current_title"] = "AI Developer"
    c["profile"]["headline"] = "AI Developer | LangChain | GPT Integrations"
    c["profile"]["summary"] = (
        "AI developer who builds LLM-powered applications using LangChain and "
        "the OpenAI API. Specialised in prompt engineering and GPT integrations."
    )
    c["career_history"] = [{
        "company": "StartupX", "title": "AI Developer",
        "start_date": "2024-01-01", "end_date": None,
        "duration_months": 17, "is_current": True, "industry": "Software",
        "company_size": "11-50",
        "description": (
            "Built internal Q&A chatbots using LangChain and OpenAI GPT-4. "
            "Integrated LangChain document loaders with S3. No traditional ML, "
            "no retrieval systems, no search infrastructure."
        ),
    }]
    c["skills"] = [
        {"name": "LangChain", "proficiency": "advanced", "endorsements": 5, "duration_months": 14},
        {"name": "OpenAI", "proficiency": "advanced", "endorsements": 3, "duration_months": 14},
    ]
    result = structural_score(c)
    assert "langchain_only" in result.penalties, (
        f"Expected langchain_only penalty, got: {result.penalties}"
    )


def test_irrelevant_career_hard_floor():
    """Civil Engineers must be capped far below a real ML engineer
    regardless of YOE and logistics."""
    civil = make_candidate(cid="CAND_0000080")
    civil["profile"]["current_title"] = "Civil Engineer"
    civil["profile"]["headline"] = "Civil Engineer | Structural Design"
    civil["profile"]["summary"] = "Structural engineer with 8 years designing bridges and roads."
    civil["career_history"] = [{
        "company": "InfraCo", "title": "Civil Engineer",
        "start_date": "2018-01-01", "end_date": None,
        "duration_months": 96, "is_current": True, "industry": "Construction",
        "company_size": "201-500",
        "description": "Designed structural load calculations for highway bridges.",
    }]
    civil["skills"] = [
        {"name": "AutoCAD", "proficiency": "expert", "endorsements": 20, "duration_months": 80},
        {"name": "STAAD Pro", "proficiency": "advanced", "endorsements": 10, "duration_months": 60},
    ]
    civil["profile"]["years_of_experience"] = 8.0
    ml_engineer = make_candidate(cid="CAND_0000081")
    ml_engineer["profile"]["country"] = "Canada"  # abroad penalty applied
    r_civil = structural_score(civil)
    r_ml = structural_score(ml_engineer)
    assert r_civil.score < r_ml.score, (
        f"Civil ({r_civil.score:.3f}) must score below abroad ML engineer ({r_ml.score:.3f})"
    )


# ---------------------------------------------------------------------------
# Structural / JD disqualifiers
# ---------------------------------------------------------------------------

def test_keyword_stuffer_is_crushed():
    """The sample_submission trap: HR Manager with a perfect AI skill list."""
    stuffer = make_candidate(
        cid="CAND_0000002",
        profile={"current_title": "HR Manager",
                 "headline": "HR Manager",
                 "summary": "HR professional managing recruitment cycles and payroll."},
    )
    for job in stuffer["career_history"]:
        job["title"] = "HR Manager"
        job["description"] = "Managed end-to-end recruitment, payroll and employee relations."
    stuffer["skills"] = [
        {"name": n, "proficiency": "expert", "endorsements": 10, "duration_months": 36}
        for n in ("RAG", "Pinecone", "Embeddings", "LLM", "PyTorch", "NLP")
    ]
    real = structural_score(make_candidate())
    fake = structural_score(stuffer)
    assert "keyword_stuffer" in fake.penalties
    assert fake.score < real.score * 0.2


def test_consulting_only_career_is_penalized_but_mixed_is_not():
    consulting = make_candidate(cid="CAND_0000003")
    for job in consulting["career_history"]:
        job["industry"] = "IT Services"
    assert "consulting_only" in structural_score(consulting).penalties

    mixed = make_candidate(cid="CAND_0000004")
    mixed["career_history"][1]["industry"] = "IT Services"  # prior services, current product
    assert "consulting_only" not in structural_score(mixed).penalties


def test_title_chaser_is_penalized():
    c = make_candidate(cid="CAND_0000005")
    c["career_history"] = [
        {
            "company": f"Hop{i}", "title": t, "start_date": f"{2020+i}-01-01",
            "end_date": None if i == 5 else f"{2020+i}-12-01",
            "duration_months": 12, "is_current": i == 5,
            "industry": "Software", "company_size": "51-200",
            "description": "Built ranking systems in production.",
        }
        for i, t in enumerate(
            ["Engineer", "Senior Engineer", "Staff Engineer", "Senior Staff", "Principal", "Principal"], 1
        )
    ]
    assert "title_chaser" in structural_score(c).penalties


def test_plain_language_strong_candidate_scores_well():
    """The inverse trap: no buzzwords in skills, but real systems in history."""
    c = make_candidate(cid="CAND_0000006")
    c["skills"] = [
        {"name": "Java", "proficiency": "advanced", "endorsements": 15, "duration_months": 60},
    ]
    result = structural_score(c)
    assert result.score > 0.5  # career evidence carries it
    assert result.components["career_evidence"] > 0.5


# ---------------------------------------------------------------------------
# Behavioral
# ---------------------------------------------------------------------------

def test_ghost_candidate_is_downweighted():
    ghost = make_candidate(redrob_signals={
        "last_active_date": "2025-10-01",   # ~8 months before REFERENCE_DATE
        "recruiter_response_rate": 0.05,
        "open_to_work_flag": False,
    })
    active = make_candidate()
    assert behavioral_multiplier(ghost).multiplier < 0.45
    assert behavioral_multiplier(active).multiplier > behavioral_multiplier(ghost).multiplier


# ---------------------------------------------------------------------------
# End-to-end: selection, tie-break, CSV format
# ---------------------------------------------------------------------------

def test_tiebreak_is_candidate_id_ascending(tmp_path: Path):
    a = make_candidate(cid="CAND_0000010")
    b = make_candidate(cid="CAND_0000002")  # identical profile, smaller id
    lookup = {"CAND_0000010": 0.5, "CAND_0000002": 0.5}.get
    ranked = select_top(iter([a, b]), lookup, top_k=2)
    out = tmp_path / "team_test.csv"
    write_submission(ranked, out)

    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["candidate_id"] == "CAND_0000002"
    assert rows[0]["score"] == rows[1]["score"]


def test_submission_csv_format(tmp_path: Path):
    candidates = [make_candidate(cid=f"CAND_{i:07d}") for i in range(1, 21)]
    lookup = {c["candidate_id"]: 0.5 + i * 0.01 for i, c in enumerate(candidates)}.get
    ranked = select_top(iter(candidates), lookup, top_k=20)
    out = tmp_path / "team_test.csv"
    write_submission(ranked, out)

    with open(out, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    assert header == ["candidate_id", "rank", "score", "reasoning"]
    assert [int(r[1]) for r in rows] == list(range(1, 21))
    scores = [float(r[2]) for r in rows]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert all(r[3].strip() for r in rows)  # reasoning never empty


def test_reasoning_varies_and_matches_tone(tmp_path: Path):
    candidates = [make_candidate(cid=f"CAND_{i:07d}") for i in range(1, 13)]
    lookup = {c["candidate_id"]: 0.9 - i * 0.05 for i, c in enumerate(candidates)}.get
    ranked = select_top(iter(candidates), lookup, top_k=12)
    out = tmp_path / "team_test.csv"
    write_submission(ranked, out)
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    reasonings = [r["reasoning"] for r in rows]
    assert len(set(reasonings)) == len(reasonings)  # all distinct


def test_honeypot_never_outranks_clean_equivalent():
    clean = make_candidate(cid="CAND_0000020")
    honeypot = make_candidate(cid="CAND_0000021", profile={"years_of_experience": 9.0})
    honeypot["career_history"] = [honeypot["career_history"][0]]
    honeypot["career_history"][0]["start_date"] = "2024-09-01"
    honeypot["career_history"][0]["duration_months"] = 21
    honeypot["skills"] = [
        {"name": n, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for n in ("RAG", "Pinecone", "Embeddings", "LLM")
    ]
    s_clean = score_candidate(clean, 0.8)
    s_honey = score_candidate(honeypot, 0.95)  # even with higher semantic sim
    assert s_clean.final > s_honey.final * 5


# ---------------------------------------------------------------------------
# New tests: recruiter-revealed signals, tier-1 prestige, research-title penalty
# ---------------------------------------------------------------------------

def test_recruiter_saves_boost_behavioral():
    """Candidates saved by recruiters score higher than those with zero saves."""
    no_saves = make_candidate(cid="CAND_0001001")
    no_saves["redrob_signals"]["saved_by_recruiters_30d"] = 0
    five_saves = make_candidate(cid="CAND_0001002")
    five_saves["redrob_signals"]["saved_by_recruiters_30d"] = 5
    # Disable other positive signals so we stay below BEHAVIORAL_CEILING
    for c in (no_saves, five_saves):
        c["redrob_signals"]["open_to_work_flag"] = False
        c["redrob_signals"]["github_activity_score"] = None
        c["redrob_signals"]["verified_email"] = False
        c["redrob_signals"]["verified_phone"] = False
    m_zero = behavioral_multiplier(no_saves).multiplier
    m_five = behavioral_multiplier(five_saves).multiplier
    assert m_five > m_zero, f"saves boost failed: {m_zero:.4f} vs {m_five:.4f}"


def test_active_applicant_boosted():
    """Active applicants (applications_submitted_30d > 0) score above passive ones."""
    signals_base = {
        "last_active_date": "2026-05-15",
        "recruiter_response_rate": 0.90,
        "open_to_work_flag": False,
        "interview_completion_rate": None,
        "github_activity_score": None,
        "verified_email": False,
        "verified_phone": False,
        "saved_by_recruiters_30d": 0,
        "profile_views_received_30d": 0,
        "notice_period_days": 30,
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
    }
    passive = make_candidate(cid="CAND_0001003", redrob_signals={**signals_base, "applications_submitted_30d": 0})
    active = make_candidate(cid="CAND_0001004", redrob_signals={**signals_base, "applications_submitted_30d": 3})
    m_passive = behavioral_multiplier(passive).multiplier
    m_active = behavioral_multiplier(active).multiplier
    assert m_active > m_passive, f"active applicant not boosted: {m_passive:.4f} vs {m_active:.4f}"


def test_tier1_company_prestige_bonus():
    """Candidate with Tier-1 product company (Google) should outscore identical no-Tier-1 candidate."""
    no_tier1 = make_candidate(cid="CAND_0001005")
    tier1_cand = make_candidate(cid="CAND_0001006")
    tier1_cand["career_history"][1]["company"] = "Google"
    r_base = structural_score(no_tier1)
    r_tier1 = structural_score(tier1_cand)
    assert r_tier1.score > r_base.score, (
        f"Tier-1 bonus not applied: base={r_base.score:.4f} tier1={r_tier1.score:.4f}"
    )
    assert any("Tier-1" in e for e in r_tier1.evidence), (
        f"Expected Tier-1 in evidence but got: {r_tier1.evidence}"
    )


def test_research_title_no_prod_is_penalized():
    """AI Research Engineer with no production-deployment vocabulary => penalty.
    Overrides profile summary and headline to strip base fixture production text."""
    c = make_candidate(cid="CAND_0001007")
    c["profile"]["current_title"] = "AI Research Engineer"
    c["profile"]["summary"] = "Researching NLP model architectures and evaluation methods."
    c["profile"]["headline"] = "AI Research Engineer | NLP"
    c["career_history"] = [{
        "company": "ProductStartup", "title": "AI Research Engineer",
        "start_date": "2022-01-01", "end_date": None, "duration_months": 28,
        "is_current": True, "industry": "Software", "company_size": "51-200",
        "description": (
            "Developed NLP models for text classification. Explored transformer "
            "architectures, trained models on proprietary datasets, published reports."
        ),
    }]
    result = structural_score(c)
    assert "research_title_no_prod" in result.penalties, (
        f"Expected research_title_no_prod penalty, got: {result.penalties}"
    )


def test_research_title_with_prod_is_not_penalized():
    """AI Research Engineer with strong production-deployment evidence => no penalty."""
    c = make_candidate(cid="CAND_0001008")
    c["profile"]["current_title"] = "AI Research Engineer"
    c["profile"]["summary"] = "Engineer who builds and ships ML ranking systems to production."
    c["profile"]["headline"] = "AI Research Engineer | Production ML"
    c["career_history"] = [{
        "company": "ProductCo", "title": "AI Research Engineer",
        "start_date": "2022-01-01", "end_date": None, "duration_months": 28,
        "is_current": True, "industry": "Software", "company_size": "201-500",
        "description": (
            "Built and deployed NLP ranking systems to production serving 2M users. "
            "Shipped a hybrid retrieval pipeline; monitored latency and A/B tested "
            "recall vs precision. Responsible for full production inference stack."
        ),
    }]
    result = structural_score(c)
    assert "research_title_no_prod" not in result.penalties, (
        f"research_title_no_prod incorrectly fired: {result.penalties}"
    )


def test_title_evidence_phrases_rotate():
    """Different ML titles should yield at least 2 distinct evidence phrase templates."""
    from ranker.structural import _title_domain_score, StructuralResult
    titles = [
        "Machine Learning Engineer", "Search Engineer", "NLP Engineer",
        "AI Engineer", "Applied Scientist",
    ]
    seen_phrases = set()
    for title in titles:
        r = StructuralResult()
        _title_domain_score({"current_title": title, "headline": ""}, r)
        if r.evidence:
            seen_phrases.add(r.evidence[0])
    assert len(seen_phrases) >= 2, f"phrases not rotating: {seen_phrases}"


def test_heap_replacement_correctness():
    """Verify that select_top correctly evicts low scores and keeps the highest ones
    when the candidate pool size exceeds the heap limit (top_k * pool_factor)."""
    # Create 350 candidates (limit is 100 * 3 = 300).
    # The first 300 candidates will have low semantic scores (e.g. 0.1).
    # The last 50 candidates will have high semantic scores (e.g. 0.9).
    # The selection must correctly identify the high-scoring candidates.
    candidates = []
    lookup_dict = {}
    for i in range(1, 351):
        cid = f"CAND_{i:07d}"
        candidates.append(make_candidate(cid=cid))
        if i <= 300:
            lookup_dict[cid] = 0.1
        else:
            lookup_dict[cid] = 0.9

    ranked = select_top(iter(candidates), lookup_dict.get, top_k=10)
    # The top 10 should all be from the last 50 candidates (CAND_0000301 to CAND_0000350)
    # with high semantic scores.
    assert len(ranked) == 10
    for sc in ranked:
        assert int(sc.candidate_id.split("_")[1]) > 300
        assert sc.semantic == 0.9


def test_behavioral_exponential_decay():
    """Verify that last active date uses exponential decay using config lambda."""
    from ranker.behavioral import behavioral_multiplier
    c_active = make_candidate()
    c_active["redrob_signals"]["last_active_date"] = "2026-05-30"  # 2 days inactive
    c_inactive = make_candidate()
    c_inactive["redrob_signals"]["last_active_date"] = "2026-03-01"  # 92 days inactive
    
    m_active = behavioral_multiplier(c_active)
    m_inactive = behavioral_multiplier(c_inactive)
    
    assert m_active.multiplier > m_inactive.multiplier


def test_semantic_structural_contradiction_check():
    """Verify that a high structural score combined with a low semantic score triggers a contradiction check flag."""
    c = make_candidate()
    # High structural score candidate (perfect fit candidate) scored with a very low semantic similarity (e.g. 0.05)
    sc = score_candidate(c, 0.05)
    
    # structural score is high (~0.9+), semantic is 0.05, delta > 0.45.
    # Should trigger contradiction flag and apply severe multiplier.
    assert any("semantic-structural contradiction" in flag for flag in sc.integrity.hard_flags)
    assert sc.integrity.multiplier == 0.10  # single hard flag multiplier

