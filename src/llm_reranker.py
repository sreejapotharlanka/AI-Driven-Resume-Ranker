"""
Stage 3 (optional): LLM re-ranking.

The hybrid score (embeddings + structured features) gets us a strong,
explainable shortlist fast and cheaply across the *entire* dataset (no LLM
calls needed for 10k+ resumes). For the final TOP_N candidates only, we
optionally ask Claude to act as a senior recruiter and re-rank with full
reasoning — catching nuance hybrid scoring can miss (career trajectory,
narrative consistency, culture/role fit, red flags like frequent job
hopping without growth, etc.).

This keeps cost/runtime low (only top N go through the LLM) while still
getting LLM-quality judgment where it matters most: the final shortlist.
"""

import os
import json
import pandas as pd
from src import config


def llm_rerank(df_top: pd.DataFrame, jd_text: str) -> pd.DataFrame:
    if not os.getenv("ANTHROPIC_API_KEY"):
        df_top = df_top.copy()
        df_top["llm_reasoning"] = "LLM re-rank skipped (no ANTHROPIC_API_KEY set)."
        df_top["final_score"] = df_top["hybrid_score"]
        return df_top

    import anthropic
    client = anthropic.Anthropic()

    candidates_payload = []
    for _, row in df_top.iterrows():
        candidates_payload.append({
            "candidate_id": str(row["candidate_id"]),
            "name": row.get("name", ""),
            "hybrid_score": row["hybrid_score"],
            "resume_snippet": str(row["full_text"])[:1200],
        })

    prompt = f"""You are a senior technical recruiter. Re-rank the following
pre-shortlisted candidates against this job description, using your judgment
about real fit (career trajectory, depth vs breadth, narrative consistency,
ownership/impact, any red flags). The hybrid_score given was computed by an
algorithm and is a starting point, not ground truth — feel free to disagree.

JOB DESCRIPTION:
{jd_text}

CANDIDATES (JSON):
{json.dumps(candidates_payload, indent=2)}

Return ONLY a JSON array, no markdown fences, no preamble, with this schema:
[
  {{"candidate_id": string, "final_score": number (0-100), "reasoning": string (1-2 sentences)}}
]
Order does not matter in the array; final_score is what determines rank.
"""

    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    raw = raw.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()

    try:
        results = json.loads(raw)
        score_map = {r["candidate_id"]: r["final_score"] for r in results}
        reason_map = {r["candidate_id"]: r["reasoning"] for r in results}
    except Exception:
        df_top = df_top.copy()
        df_top["llm_reasoning"] = "LLM re-rank failed to parse; falling back to hybrid score."
        df_top["final_score"] = df_top["hybrid_score"]
        return df_top.sort_values("final_score", ascending=False).reset_index(drop=True)

    df_top = df_top.copy()
    df_top["final_score"] = df_top["candidate_id"].astype(str).map(score_map).fillna(df_top["hybrid_score"])
    df_top["llm_reasoning"] = df_top["candidate_id"].astype(str).map(reason_map).fillna("")
    return df_top.sort_values("final_score", ascending=False).reset_index(drop=True)
