"""
Extracts scoring features from the rich JSONL candidate data.
Uses the nested skills, career history, and redrob_signals properly.
"""

import re
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from src import config

BEHAVIORAL_PHRASES = [
    "led", "managed", "owned", "drove", "spearheaded", "launched", "scaled",
    "improved", "increased", "reduced", "optimized", "delivered", "architected",
    "responsible for", "cross-functional", "stakeholder", "end-to-end",
    "mentored", "initiated", "presented", "collaborated", "shipped",
]


def skill_match_score(skill_names: list, required_skills: list) -> tuple:
    """Returns (score 0-1, count of matched skills)."""
    if not required_skills:
        return 0.5, 0
    skill_lower = [s.lower() for s in skill_names]
    matched = sum(
        1 for req in required_skills
        if any(req.lower() in s or s in req.lower() for s in skill_lower)
    )
    return matched / len(required_skills), matched


def experience_match_score(years: float, min_years: float, max_years) -> float:
    if years >= min_years:
        if max_years and years <= max_years + 2:
            return 1.0
        elif max_years and years > max_years + 2:
            return max(0.6, 1.0 - (years - max_years - 2) * 0.04)
        return 1.0
    gap = min_years - years
    return max(0.0, 1.0 - gap * 0.18)


def behavioral_signal_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for p in BEHAVIORAL_PHRASES if p in text_lower)
    return min(1.0, hits / 8.0)


def platform_signal_score(row) -> float:
    """Composite redrob platform engagement score."""
    score = 0.0
    # Response rate (very important — shows engagement)
    score += row["recruiter_response_rate"] * 0.30
    # Profile completeness
    score += (row["profile_completeness"] / 100.0) * 0.20
    # GitHub activity
    score += (row["github_activity_score"] / 100.0) * 0.15
    # Avg skill assessment score
    score += (row["avg_assessment_score"] / 100.0) * 0.15
    # Interview completion
    score += row["interview_completion_rate"] * 0.10
    # Open to work bonus
    score += 0.05 if row["open_to_work"] else 0.0
    # LinkedIn connected bonus
    score += 0.03 if row["linkedin_connected"] else 0.0
    # Saved by recruiters (normalized, cap at 20)
    score += min(row["saved_by_recruiters_30d"] / 20.0, 1.0) * 0.07
    return min(1.0, score)


def calculate_triage_coefficient(row) -> tuple:
    """
    Returns (coeff: float, reason: str)
    coeff is 0.0 for hard-disqualified, < 1.0 for down-weighted, 1.0 for fully qualified.
    """
    title = str(row["current_title"]).lower()
    
    # 1. Role Title relevance check
    for disq in config.DISQUALIFIED_ROLES:
        if disq in title:
            return 0.0, f"Disqualified role title: '{row['current_title']}'"
            
    # 2. Consulting-only career check
    if row["only_consulting"]:
        return 0.0, "Career exclusively at consulting/services firms"
        
    # 3. Title Chaser check (switching too fast)
    # Check average tenure in months for candidates with > 1.5 years experience
    if row["average_tenure_months"] > 0 and row["average_tenure_months"] < config.MIN_AVG_TENURE_MONTHS:
        if row["years_of_experience"] > 1.5:
            return 0.0, f"Title chaser: frequent job switcher (avg tenure {row['average_tenure_months']:.1f} months)"
            
    # 4. Location & Visa Check
    country = str(row.get("country", "")).lower()
    location = str(row.get("location", "")).lower()
    willing_to_relocate = row.get("willing_to_relocate", False)
    
    if country and country != "india" and not willing_to_relocate:
        return 0.0, "Located outside India and unwilling to relocate (no visa sponsorship)"
        
    # 5. Inactivity Down-weighting & Preference Boost
    coeff = 1.0
    reasons = []
    
    if row["days_inactive"] > 180 and row["recruiter_response_rate"] <= 0.05:
        coeff *= 0.05
        reasons.append("Highly inactive: inactive > 6 months, response rate <= 5%")
    elif row["days_inactive"] > 180:
        coeff *= 0.5
        reasons.append("Inactive: inactive > 6 months")
    elif row["recruiter_response_rate"] <= 0.05:
        coeff *= 0.3
        reasons.append("Low responsiveness: response rate <= 5%")
        
    # Location preference check
    is_target_location = any(loc in location for loc in config.TARGET_LOCATIONS)
    if not is_target_location and not willing_to_relocate:
        coeff *= 0.8
        reasons.append("Not in target location & unwilling to relocate")
    elif is_target_location:
        coeff *= 1.05  # subtle boost for local candidates
        reasons.append("Preferred location match")
        
    reason_str = "; ".join(reasons) if reasons else "Fully qualified"
    return min(1.1, coeff), reason_str


def build_features(df: pd.DataFrame, jd_spec: dict) -> pd.DataFrame:
    df = df.copy()
    required_skills = jd_spec.get("required_skills", [])
    min_exp = jd_spec.get("min_experience_years", 0) or 0
    max_exp = jd_spec.get("max_experience_years", None)

    results = df["skill_names"].apply(
        lambda names: skill_match_score(names, required_skills)
    )
    df["skill_match"] = results.apply(lambda x: x[0])
    df["matched_skill_count"] = results.apply(lambda x: x[1])

    df["experience_match"] = df["years_of_experience"].apply(
        lambda y: experience_match_score(y, min_exp, max_exp)
    )
    df["behavioral_signal"] = df["career_text"].apply(behavioral_signal_score)
    df["platform_score"] = df.apply(platform_signal_score, axis=1)

    # Compute triage coefficient and reason
    triage_results = df.apply(calculate_triage_coefficient, axis=1)
    df["triage_coeff"] = triage_results.apply(lambda x: x[0])
    df["triage_reason"] = triage_results.apply(lambda x: x[1])

    return df

