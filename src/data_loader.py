"""
Loads candidates.jsonl and flattens the rich nested schema into a DataFrame.
Also handles the job_description.docx file.
"""

import json
import os
import pandas as pd
from src import config


def load_candidates(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(config.DATA_DIR, "candidates.jsonl")
    if not os.path.exists(path):
        # fallback to sample
        path = os.path.join(config.SAMPLE_DATA_DIR, "sample_candidates.json")

    ext = os.path.splitext(path)[1].lower()
    records = []

    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported format: {ext}")

    rows = []
    from datetime import datetime
    ref_date = datetime(2026, 7, 2)  # Reference date matching current conversation time

    for c in records:
        p = c.get("profile", {})
        rs = c.get("redrob_signals", {})

        # Flatten career history into one text blob
        career_texts = []
        career_history = c.get("career_history", [])
        companies = [job.get("company", "").lower().strip() for job in career_history]
        
        # Check if they have ONLY worked at consulting firms
        only_consulting = False
        if companies:
            only_consulting = all(
                any(cf in comp for cf in config.CONSULTING_FIRMS)
                for comp in companies
            )

        # Average tenure in months (ignoring jobs with 0 duration)
        durations = [job.get("duration_months", 0) for job in career_history if job.get("duration_months", 0) > 0]
        avg_tenure = sum(durations) / len(durations) if durations else 0.0

        # Inactivity calculation (days since last active)
        last_active_str = rs.get("last_active_date", "")
        days_inactive = 0
        if last_active_str:
            try:
                active_date = datetime.strptime(last_active_str, "%Y-%m-%d")
                days_inactive = max(0, (ref_date - active_date).days)
            except ValueError:
                days_inactive = 0

        for job in career_history:
            career_texts.append(
                f"{job.get('title','')} at {job.get('company','')} "
                f"({job.get('industry','')}): {job.get('description','')}"
            )
        career_text = " | ".join(career_texts)

        # Skills list and weighted skill text
        skills_objs = c.get("skills", [])
        skill_names = [s["name"] for s in skills_objs]
        # Weight expert/advanced skills more in text
        skill_text_parts = []
        for s in skills_objs:
            weight = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", "beginner"), 1)
            skill_text_parts.extend([s["name"]] * weight)
        skill_text = " ".join(skill_text_parts)

        # Education text
        edu_parts = []
        for e in c.get("education", []):
            edu_parts.append(
                f"{e.get('degree','')} in {e.get('field_of_study','')} "
                f"from {e.get('institution','')} (tier: {e.get('tier','unknown')})"
            )
        edu_text = " ".join(edu_parts)

        # Certifications
        cert_text = " ".join(
            cert.get("name", "") for cert in c.get("certifications", [])
        )

        # Skill assessment scores avg
        assessments = rs.get("skill_assessment_scores", {})
        avg_assessment = (
            sum(assessments.values()) / len(assessments) if assessments else 0
        )

        # Full text for embeddings
        full_text = " ".join(filter(None, [
            p.get("headline", ""),
            p.get("summary", ""),
            career_text,
            skill_text,
            edu_text,
            cert_text,
        ]))

        rows.append({
            "candidate_id": c["candidate_id"],
            "name": p.get("anonymized_name", ""),
            "current_title": p.get("current_title", ""),
            "headline": p.get("headline", ""),
            "summary": p.get("summary", ""),
            "location": p.get("location", ""),
            "country": p.get("country", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "current_industry": p.get("current_industry", ""),
            "current_company_size": p.get("current_company_size", ""),
            "career_text": career_text,
            "skill_names": skill_names,
            "skill_text": skill_text,
            "edu_text": edu_text,
            "full_text": full_text,
            # parsed career history details
            "only_consulting": only_consulting,
            "average_tenure_months": avg_tenure,
            "days_inactive": days_inactive,
            # redrob signals
            "profile_completeness": rs.get("profile_completeness_score", 0),
            "open_to_work": rs.get("open_to_work_flag", False),
            "recruiter_response_rate": rs.get("recruiter_response_rate", 0),
            "github_activity_score": max(0, rs.get("github_activity_score", 0)),
            "interview_completion_rate": rs.get("interview_completion_rate", 0),
            "offer_acceptance_rate": max(0, rs.get("offer_acceptance_rate", 0)),
            "connection_count": rs.get("connection_count", 0),
            "endorsements_received": rs.get("endorsements_received", 0),
            "avg_assessment_score": avg_assessment,
            "saved_by_recruiters_30d": rs.get("saved_by_recruiters_30d", 0),
            "notice_period_days": rs.get("notice_period_days", 30),
            "willing_to_relocate": rs.get("willing_to_relocate", False),
            "linkedin_connected": rs.get("linkedin_connected", False),
            "verified_email": rs.get("verified_email", False),
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} candidates.")
    return df


def load_job_description(path: str = None) -> str:
    if path is None:
        # Try docx first, then txt
        docx_path = os.path.join(config.DATA_DIR, "job_description.docx")
        txt_path = os.path.join(config.SAMPLE_DATA_DIR, "job_description.txt")
        if os.path.exists(docx_path):
            path = docx_path
        else:
            path = txt_path

    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        # pyrefly: ignore [missing-import]
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
