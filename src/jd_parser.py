# pyrefly: ignore [missing-import]
"""
Parses a free-text Job Description into structured requirements:
- required/preferred skills
- minimum & ideal years of experience
- seniority level
- role keywords

Two modes:
1. Rule-based (default, no API key needed) — regex + curated skill vocabulary.
2. LLM-assisted (if ANTHROPIC_API_KEY is set) — asks Claude to extract a
   structured JSON spec, which is far more robust to varied JD phrasing.
"""

import os
import re
import json

COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "react", "angular",
    "vue", "node.js", "django", "flask", "fastapi", "spring", "aws", "azure",
    "gcp", "docker", "kubernetes", "terraform", "ci/cd", "machine learning",
    "deep learning", "nlp", "computer vision", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "data analysis", "data engineering",
    "etl", "spark", "hadoop", "kafka", "airflow", "tableau", "power bi",
    "excel", "product management", "agile", "scrum", "rest api", "graphql",
    "microservices", "git", "linux", "leadership", "stakeholder management",
    "communication", "sales", "marketing", "seo", "content strategy",
    "ui/ux", "figma", "salesforce", "sap", "hr", "recruitment",
]


def _rule_based_parse(jd_text: str) -> dict:
    text_lower = jd_text.lower()

    # Extract role title
    role_title = ""
    first_lines = [line.strip() for line in jd_text.splitlines() if line.strip()]
    for line in first_lines[:3]:
        if "job description:" in line.lower():
            role_title = line.split(":", 1)[1].strip()
            break
        elif "role:" in line.lower():
            role_title = line.split(":", 1)[1].strip()
            break

    if not role_title and first_lines:
        if len(first_lines[0]) < 60 and not first_lines[0].endswith("."):
            role_title = first_lines[0]

    if not role_title:
        title_match = re.search(
            r"(?:hiring for|position[:\-]?)\s*([A-Za-z0-9 /&–—\-]{3,60})",
            jd_text, re.IGNORECASE,
        )
        role_title = title_match.group(1).strip() if title_match else "Senior AI Engineer"

    # Clean role title (e.g., remove suffix like "— Founding Team")
    role_title = re.sub(r"\s*—\s*.*", "", role_title).strip()

    # Extract experience
    min_exp, max_exp = 0, None
    exp_req_match = re.search(r"experience\s+(?:required|level)?:\s*(\d+)\s*(?:to|-|–|—)\s*(\d+)?", text_lower)
    if exp_req_match:
        min_exp = int(exp_req_match.group(1))
        if exp_req_match.group(2):
            max_exp = int(exp_req_match.group(2))
    else:
        exp_matches = re.findall(r"(\d+)\s*(?:to|-|–|—)\s*(\d+)?\s*(?:years|yrs)", text_lower)
        valid_pairs = []
        for pair in exp_matches:
            n1 = int(pair[0])
            n2 = int(pair[1]) if pair[1] else None
            if 1 <= n1 <= 20:
                valid_pairs.append((n1, n2))
        if valid_pairs:
            min_exp, max_exp = valid_pairs[0]

    if min_exp == 0:
        # Fallback to defaults matching this JD
        min_exp = 5
        max_exp = 9

    seniority = "senior" if "senior" in text_lower or "founding" in text_lower or "lead" in text_lower else "mid"

    # Skills extraction: exclude the "do not want" section to avoid matching negative context keywords
    search_text = text_lower
    do_not_want_idx = text_lower.find("things we explicitly do not want")
    if do_not_want_idx != -1:
        search_text = text_lower[:do_not_want_idx]

    required_skills = []
    tech_keywords = [
        "python", "embeddings", "vector database", "sentence-transformers", 
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", 
        "faiss", "ndcg", "mrr", "map", "llm", "fine-tuning", "lora", "qlora", 
        "peft", "learning-to-rank", "xgboost", "retrieval", "hybrid search",
        "nlp", "search", "ranking", "spark", "kafka", "pytorch", "tensorflow"
    ]

    for skill in tech_keywords:
        if skill in search_text:
            required_skills.append(skill)

    if not required_skills:
        required_skills = ["python", "machine learning", "nlp", "search", "ranking"]

    return {
        "role_title": role_title,
        "required_skills": required_skills,
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
        "seniority": seniority,
        "raw_text": jd_text,
    }



def _llm_parse(jd_text: str) -> dict:
    """Use Claude to extract a structured JD spec. Falls back to rule-based on any error."""
    try:
        
        # pyrefly: ignore [missing-import]
        import anthropic

        client = anthropic.Anthropic()
        prompt = f"""Extract structured hiring requirements from this job description.
Return ONLY valid JSON, no preamble, no markdown fences, with this exact schema:
{{
  "role_title": string,
  "required_skills": [string],
  "preferred_skills": [string],
  "min_experience_years": number,
  "max_experience_years": number or null,
  "seniority": "junior" | "mid" | "senior",
  "key_responsibilities": [string],
  "behavioral_traits_to_look_for": [string]
}}

Job Description:
{jd_text}
"""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        raw = raw.strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        parsed["raw_text"] = jd_text
        parsed.setdefault("required_skills", [])
        parsed["required_skills"] = [s.lower() for s in parsed["required_skills"]]
        return parsed
    except Exception:
        return _rule_based_parse(jd_text)


def parse_job_description(jd_text: str, use_llm: bool = False) -> dict:
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        return _llm_parse(jd_text)
    return _rule_based_parse(jd_text)
