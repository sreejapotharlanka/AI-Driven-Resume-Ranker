import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, "sample_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")

# Hybrid score weights — must sum to 1.0
SCORE_WEIGHTS = {
    "semantic_similarity": 0.35,
    "skill_match":         0.30,
    "experience_match":    0.15,
    "behavioral_signal":   0.10,
    "platform_activity":   0.10,
}

USE_LLM_RERANK = os.getenv("USE_LLM_RERANK", "false").lower() == "true"
TOP_N_FOR_LLM_RERANK = int(os.getenv("TOP_N_FOR_LLM_RERANK", 20))
ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"

# Recruiter Triage Filters
DISQUALIFIED_ROLES = [
    "accountant", "graphic designer", "project manager", "operations manager", 
    "customer support", "hr manager", "human resources", "recruiter", 
    "content writer", "civil engineer", "mechanical engineer", "sales", 
    "marketing", "product manager", "scrum master", "business analyst", 
    "qa engineer", "tester", "ui/ux", "designer", "architect", "business development"
]

CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", 
    "cognizant", "capgemini", "tech mahindra", "hcl", "cognizant technology",
    "cap gemini", "tata consulting"
]

# Average tenure (in months) threshold for title chasers
MIN_AVG_TENURE_MONTHS = 18.0

# Priority locations in India (Tier-1 and preferred office locations)
TARGET_LOCATIONS = ["pune", "noida", "delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru", "chennai"]

