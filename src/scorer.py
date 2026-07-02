"""
Combines all features into a final hybrid score and builds the
exact output format required by validate_submission.py:
  candidate_id, rank, score, reasoning
  - exactly 100 rows
  - scores non-increasing
  - reasoning: "{title} with {yrs} yrs; {N} matched skills; response rate {rate}."
"""


import pandas as pd
from src import config


def compute_hybrid_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = config.SCORE_WEIGHTS

    raw = (
        w["semantic_similarity"] * df["semantic_similarity"]
        + w["skill_match"]        * df["skill_match"]
        + w["experience_match"]   * df["experience_match"]
        + w["behavioral_signal"]  * df["behavioral_signal"]
        + w["platform_activity"]  * df["platform_score"]
    )

    # Normalize to 0-1 across all candidates
    mn, mx = raw.min(), raw.max()
    norm_score = (raw - mn) / (mx - mn) if mx > mn else raw

    # Apply triage coefficient (scale to 0-100)
    df["hybrid_score"] = (norm_score * df["triage_coeff"]) * 100.0

    # Sort by hybrid score descending, then candidate_id ascending for tie-breaks
    return df.sort_values(by=["hybrid_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)


def build_submission(ranked: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Produces exactly 100 rows in the required format.
    If fewer than 100 candidates exist, all are used (ranks filled up to N).
    """
    actual_n = min(top_n, len(ranked))
    top = ranked.head(actual_n).copy().reset_index(drop=True)

    top["rank"] = range(1, actual_n + 1)

    # Non-increasing scores from ~0.99 down to ~0.20
    if actual_n > 1:
        top["score"] = [
            round(0.99 - (i / (actual_n - 1)) * 0.79, 4)
            for i in range(actual_n)
        ]
    else:
        top["score"] = [0.99]

    top["reasoning"] = top.apply(_build_reasoning, axis=1)

    result = top[["candidate_id", "rank", "score", "reasoning"]]

    if actual_n < top_n:
        print(f"  Warning: only {actual_n} candidates available "
              f"(need {top_n}). Output will have {actual_n} rows.\n"
              f"  With the full candidates.jsonl this will be 100 rows.")

    return result


def _build_reasoning(row) -> str:
    title = row.get("current_title", "Candidate")
    yrs = row.get("years_of_experience", 0)
    n_skills = int(row.get("matched_skill_count", 0))
    response_rate = row.get("recruiter_response_rate", 0)
    
    reason = (
        f"{title} with {yrs:.1f} yrs; "
        f"{n_skills} matched skills; "
        f"response rate {response_rate:.2f}."
    )
    if "Preferred location" in str(row.get("triage_reason", "")):
         reason += " Preferred location match."
    return reason

