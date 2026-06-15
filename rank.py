#!/usr/bin/env python3
"""Produce the top-100 submission CSV using offline rule-based scoring and heap ranking.
Satisfies constraints: CPU only, offline, fast execution.
"""

from __future__ import annotations

import argparse
import time
import json
from pathlib import Path
import pandas as pd
import heapq

from ranker.loading import load_job_description
from ranker.structural import parse_job_description
from ranker.pipeline import score_candidate, _InvertedStr, ScoredCandidate, write_submission

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidates Parquet file or raw JSONL/JSONL.gz")
    parser.add_argument("--jd", default="data/job_description.txt", help="job description text file")
    parser.add_argument("--artifacts", default="artifacts", help="ignored for backward compatibility")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    total_start = time.time()

    # -- 1. Loading candidates (loading time) -----------------------------
    started_load = time.time()
    cand_path = Path(args.candidates)
    if cand_path.suffix == ".parquet":
        candidates_df = pd.read_parquet(cand_path)
    else:
        # Fallback to load raw JSONL on the fly (for backward compatibility)
        print(f"Loading raw candidates from {cand_path} and preprocessing on the fly...")
        from precompute import process_candidate
        from ranker.loading import iter_candidates
        records = []
        for c in iter_candidates(cand_path):
            records.append(process_candidate(c))
        candidates_df = pd.DataFrame(records)
        
    loading_time = time.time() - started_load

    # Parse JD
    jd_text = load_job_description(args.jd)
    jd_info = parse_job_description(jd_text)

    # -- 2. Scoring & Heap ranking -----------------------------------------
    started_score = time.time()
    req_skills = jd_info["required_skills"]
    pref_skills = jd_info["preferred_skills"]
    min_exp = jd_info["min_experience"]
    max_exp = jd_info.get("max_experience")
    role_kws = jd_info["role_keywords"]
    
    from ranker.config import get_dynamic_weights
    weights = get_dynamic_weights(jd_info.get("jd_text", ""))
    
    heap = []
    top_k = args.top_k
    
    # Convert dataframe to dictionary records for fast iteration and push directly to heap
    for row in candidates_df.to_dict('records'):
        scored = score_candidate(row, req_skills, pref_skills, min_exp, role_kws, weights=weights, jd_max_experience=max_exp)
        entry = (scored.final, _InvertedStr(scored.candidate_id), scored)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
            
    scoring_time = time.time() - started_score

    # -- 3. Finalists sorting (ranking time) ------------------------------
    started_rank = time.time()
    finalists = sorted((e[2] for e in heap), key=ScoredCandidate.sort_key)
    ranking_time = time.time() - started_rank

    # -- 4. Output generation (output time) -------------------------------
    started_out = time.time()
    write_submission(finalists, args.out)
    output_time = time.time() - started_out

    total_time = time.time() - total_start

    # Print profiling logs as required
    print(f"Loading Time: {loading_time:.3f} sec")
    print(f"Scoring Time: {scoring_time:.3f} sec")
    print(f"Ranking Time: {ranking_time:.3f} sec")
    print(f"Total Time: {total_time:.3f} sec")

if __name__ == "__main__":
    main()
