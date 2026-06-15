"""Candidate loading and text construction shared across the pipeline.

precompute.py, rank.py and the sandbox app share helper functions from
this module to load candidate records and parse dates consistently.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
from pathlib import Path
from typing import Iterator

from . import config


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidate records from a .jsonl or .jsonl.gz file.

    Streaming keeps peak memory flat (~a few hundred MB) instead of holding
    the full 465 MB pool as Python objects.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_blob(raw: bytes) -> list[dict]:
    """Parse an uploaded blob that may be JSON array, JSONL, or gzipped JSONL.

    Used by the sandbox app, which accepts small samples in any of the
    bundle's formats.
    """
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8").strip()
    if text.startswith("["):
        data = json.load(io.StringIO(text))
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of candidate objects.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def months_between(start: dt.date, end: dt.date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def career_span_years(candidate: dict) -> float:
    """Observable career span: earliest start date to reference date."""
    starts = [parse_date(j.get("start_date")) for j in candidate.get("career_history", [])]
    starts = [s for s in starts if s]
    if not starts:
        return 0.0
    return max(0.0, months_between(min(starts), config.REFERENCE_DATE) / 12.0)






def load_job_description(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
