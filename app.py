"""Sandbox app for the Redrob India Runs Data & AI Challenge (Track 1).

Satisfies submission_spec.md section 10.5: a hosted environment where the
ranking system runs end-to-end on a small candidate sample (<= 100) and
produces a ranked CSV. Embeddings for the sample are computed live (a 100-
candidate sample embeds in seconds on CPU); the full 100K run uses the
precomputed artifacts exactly as documented in the README.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ranker import config
from ranker.loading import load_candidates_blob, load_job_description

MAX_SAMPLE = 100
JD_PATH = Path("data/job_description.txt")

st.set_page_config(page_title="redrob-ranker sandbox", page_icon="🎯", layout="wide")
st.title("redrob-ranker — sandbox")
st.caption(
    "Intelligent Candidate Discovery (India Runs, Track 1). Upload a candidate "
    f"sample (≤{MAX_SAMPLE}, JSON / JSONL / JSONL.GZ) and get the ranked CSV. "
    "Scoring is identical to the full 100K pipeline."
)




@st.cache_data
def get_jd_text() -> str:
    return load_job_description(JD_PATH)


uploaded = st.file_uploader(
    "Candidate sample", type=["json", "jsonl", "gz"],
    help="Use sample_candidates.json from the hackathon bundle, or any slice of candidates.jsonl",
)

top_k = st.slider("Rows to rank", min_value=5, max_value=MAX_SAMPLE, value=25)

if uploaded is not None:
    try:
        candidates = load_candidates_blob(uploaded.getvalue())
    except (ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not parse the file: {exc}")
        st.stop()

    if len(candidates) > MAX_SAMPLE:
        st.warning(f"{len(candidates)} candidates uploaded; sandbox caps at {MAX_SAMPLE}. Truncating.")
        candidates = candidates[:MAX_SAMPLE]
    st.write(f"Parsed **{len(candidates)}** candidates.")

    if st.button("Run ranking", type="primary"):
        with st.spinner("Scoring..."):
            from precompute import process_candidate
            records = [process_candidate(c) for c in candidates]
            df_candidates = pd.DataFrame(records)
            
            from ranker.structural import parse_job_description
            jd_info = parse_job_description(get_jd_text())
            
            from ranker.pipeline import select_top, write_submission
            ranked = select_top(df_candidates, jd_info, top_k=min(top_k, len(candidates)))
            
            out_path = Path("sandbox_output.csv")
            write_submission(ranked, out_path)
            csv_text = out_path.read_text(encoding="utf-8")

        df = pd.read_csv(io.StringIO(csv_text))
        st.subheader("Ranked output")
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        col1.metric("Candidates scored", len(candidates))
        col2.metric("Rows ranked", len(ranked))

        st.download_button(
            "Download CSV", csv_text, file_name="sandbox_ranked.csv", mime="text/csv"
        )

        with st.expander("Why these ranks? (per-candidate breakdown)"):
            for sc in ranked[:10]:
                st.markdown(
                    f"**{sc.candidate_id}** — final `{sc.final:.4f}` "
                    f"(career relevance `{sc.career_relevance:.2f}`, skills match `{sc.skills_match:.2f}`, "
                    f"experience match `{sc.experience_match:.2f}`, "
                    f"behavior score `{sc.behavior_score:.2f}`, trust score `{sc.trust_score:.2f}`)"
                )
else:
    st.info("Upload a candidate file to begin. `sample_candidates.json` from the bundle works as-is.")
