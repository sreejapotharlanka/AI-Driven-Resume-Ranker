# 🎯 AI-Driven Resume Ranking & Candidate Shortlisting System

Ranks candidates the way a great recruiter would — by understanding the role
and the candidate's whole story, not by matching keywords.

## Why this approach

Pure keyword filters fail because language varies: a candidate who "built
real-time streaming pipelines" is a perfect match for a JD asking for
"experience with streaming data systems," but they share almost no words in
common. To fix this, the system combines **three layers of understanding**
instead of one brittle keyword filter:

| Layer | What it captures | Technique |
|---|---|---|
| **Recruiter Triage & Filter** | Disqualifies/down-weights irrelevant candidates (Accountants, consulting-only, title chasers, inactive profiles, visa issues) | Custom heuristics mapped from candidate career histories and platform signals |
| **Semantic similarity** | Does the resume *mean* the same thing as the JD, even with different words? | Sentence-transformer embeddings (`all-mpnet-base-v2`) + cosine similarity, with a TF-IDF fallback if offline |
| **Structured fit** | Required skills, years of experience, seniority | JD parsed into a structured spec (rule-based or LLM-assisted), matched against extracted candidate features |
| **Behavioral & platform signal** | Ownership, leadership, impact language; GitHub/LinkedIn activity if present in the dataset | Curated phrase detection + normalized activity scores |

These are combined into one explainable **hybrid score** (configurable weights in `src/config.py`) adjusted by the triage coefficient, and — optionally — the **top N** candidates are re-ranked by Claude acting as a senior recruiter. Running the LLM only on the final shortlist keeps cost and latency low.

```
                ┌─────────────────┐
   Job          │  JD Parser       │   → role_title, required_skills,
   Description  │  (rule/LLM)      │     min_experience, seniority
                └────────┬─────────┘
                         │
   Candidate    ┌────────▼─────────┐
   Dataset  ──► │  Data Loader      │   → normalizes any CSV/XLSX schema
   (54MB)       │  (column-tolerant)│     into one canonical format
                └────────┬─────────┘
                         │
              ┌──────────┼───────────────┐
              ▼          ▼               ▼
      Semantic Sim   Structured       Behavioral /
      (embeddings)   Features         Platform Signal
              │          │               │
              └──────────┼───────────────┘
                         ▼
                 Hybrid Scorer (weighted)
                         │
                         ▼
            Top-N → Optional Claude Re-rank
                         │
                         ▼
              output/ranked_candidates.csv
```

## Project structure

```
resume-ranker/
├── README.md                  ← you are here
├── requirements.txt
├── .env.example
├── app.py                      ← optional Streamlit demo UI
├── src/
│   ├── config.py                ← column mapping, score weights, paths
│   ├── data_loader.py           ← schema-tolerant dataset loader
│   ├── jd_parser.py              ← JD → structured requirements
│   ├── feature_extractor.py     ← skill match, experience, behavioral, platform scores
│   ├── embedder.py              ← semantic similarity (sentence-transformers + TF-IDF fallback)
│   ├── scorer.py                 ← hybrid score + plain-English explanation per candidate
│   ├── llm_reranker.py           ← optional Claude re-rank of the final shortlist
│   └── main.py                   ← CLI entrypoint / pipeline orchestrator
├── sample_data/
│   ├── job_description.txt      ← example JD used for the smoke test
│   └── sample_resumes.csv       ← 10 synthetic candidates for a quick demo
├── data/                        ← put the real 54MB hackathon dataset here
├── output/
│   └── sample_ranked_candidates.csv  ← example output from the sample run
└── presentation/
    └── Resume_Ranking_System.pptx / .pdf
```

## Setup

```bash
git clone <this-repo>
cd resume-ranker
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env   # optional — only needed for LLM-assisted JD parsing / re-ranking
```

## Running it

### 1. Smoke test (bundled sample data, no setup needed)

```bash
python -m src.main
```

This uses `sample_data/job_description.txt` and `sample_data/sample_resumes.csv`
and writes results to `output/ranked_candidates.csv`.

### 2. With the real hackathon dataset

1. Drop the provided dataset file into `data/` (e.g. `data/candidates.csv`).
2. If your dataset's column names differ from the defaults, adjust
   `COLUMN_MAP` in `src/config.py` (the loader already tries many common
   aliases automatically — e.g. `Resume_str`, `resume`, `cv_text` all map to
   `resume_text`).
3. Run:

```bash
python -m src.main \
  --jd path/to/job_description.txt \
  --candidates data/candidates.csv \
  --output output/ranked_candidates.csv \
  --top 25
```

Optional flags:
- `--llm-jd` — use Claude to parse the JD instead of the rule-based parser (needs `ANTHROPIC_API_KEY`)
- `--llm-rerank` — use Claude to re-rank the final top-N shortlist with recruiter-style reasoning

### 3. Interactive demo (Streamlit)

```bash
streamlit run app.py
```

Paste a JD, upload a dataset (or use the bundled sample), and get a live
ranked shortlist with explanations.

## Output format

`output/ranked_candidates.csv` contains, per shortlisted candidate:

| Column | Meaning |
|---|---|
| `rank` | Final rank (1 = best fit) |
| `candidate_id`, `name`, `current_title`, `location` | Identity fields |
| `years_experience` | Parsed years of experience |
| `semantic_similarity` | 0–1, how well the resume's meaning matches the JD |
| `skill_match` | 0–1, fraction of required skills covered |
| `experience_match` | 0–1, how well experience fits the JD's ask |
| `behavioral_signal` | 0–1, ownership/leadership/impact language strength |
| `platform_activity_score` | 0–1, GitHub/LinkedIn/StackOverflow activity if present |
| `hybrid_score` | 0–100 weighted combination of the above |
| `final_score` | `hybrid_score`, or Claude's re-ranked score if `--llm-rerank` was used |
| `match_explanation` | Plain-English summary of why this candidate ranked where they did |
| `llm_reasoning` | Claude's reasoning, if LLM re-rank was used |

## Design decisions worth highlighting (for judging)

- **Recruiter Triage & Filter Layer.** We triage candidates using actual recruiting rules:
  - *Disqualified Titles*: Hard-excludes candidates in irrelevant career families (e.g. Accountants, Graphic Designers, HR Managers, Project Managers) to avoid keyword-stuffing traps.
  - *Consulting-only backgrounds*: Excludes candidates who have exclusively worked at IT service consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini).
  - *Title chasers*: Excludes/penalizes profiles switching companies too quickly (avg tenure < 18 months).
  - *Platform Inactivity*: Heavily down-weights candidates who have not logged in for 6 months or have a recruiter response rate of <= 5%.
  - *Location & Visa*: Boosts Noida/Pune candidates and penalizes international applicants without relocation flags.
- **No hard dependency on an LLM API key.** The core ranking pipeline (embeddings + structured scoring + triage) runs entirely offline/locally and scales to 100,000+ candidates cheaply. The LLM is an optional, surgical enhancement applied only to the final shortlist — this is both cost-efficient and avoids LLM hallucination risk being load-bearing for the entire ranking.
- **Schema-tolerant data loader.** Hackathon datasets vary in column naming; rather than hardcoding column names, the loader matches against a list of common aliases so the system survives minor dataset format changes.
- **Explainability by default.** Every candidate gets a human-readable `match_explanation` (containing their experience, matched skills, response rate, and location preferences) — recruiters can audit *why* someone ranked where they did, which builds trust in the shortlist.
- **Graceful degradation.** If sentence-transformer weights can't be downloaded (offline environment), the system automatically falls back to TF-IDF similarity rather than crashing.

## Tuning

All scoring weights live in `src/config.py` under `SCORE_WEIGHTS`. Adjust per
role family (e.g. weight `behavioral_signal` higher for leadership roles,
`skill_match` higher for highly technical/niche roles).

## Next steps / known extensions

See `CONTINUE_PROMPT.md` for a ready-to-paste prompt to hand this project to
an AI coding agent (e.g. Google Antigravity / Claude Code) to continue work —
e.g. wiring up the real 54MB dataset, adding resume PDF/DOCX parsing, or
building an evaluation harness against ground-truth hiring decisions.
