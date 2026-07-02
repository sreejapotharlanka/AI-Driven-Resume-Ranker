import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src import data_loader, jd_parser, feature_extractor, embedder, scorer
from src.llm_reranker import llm_rerank


def run_pipeline(jd_path, candidates_path, output_path, top_n,
                 use_llm_jd, use_llm_rerank, use_transformers):
    print(f"\n{'='*55}")
    print("  AI Resume Ranking & Shortlisting System")
    print(f"{'='*55}\n")

    print("[1/6] Loading job description...")
    jd_text = data_loader.load_job_description(jd_path)
    print(f"      -> {len(jd_text)} characters loaded")

    print("[2/6] Parsing job description...")
    jd_spec = jd_parser.parse_job_description(jd_text, use_llm=use_llm_jd)
    print(f"      -> role: {jd_spec.get('role_title')!r}")
    print(f"      -> skills detected ({len(jd_spec.get('required_skills',[]))}): "
          f"{jd_spec.get('required_skills', [])[:8]}")
    print(f"      -> min experience: {jd_spec.get('min_experience_years')} yrs")

    print("\n[3/6] Loading candidate dataset...")
    df = data_loader.load_candidates(candidates_path)

    print("\n[4/6] Computing semantic similarity...")
    df["semantic_similarity"] = embedder.semantic_similarity(
        jd_text, df["full_text"].tolist(),
        use_transformers=use_transformers
    )

    print("\n[5/6] Extracting features & computing hybrid score...")
    df = feature_extractor.build_features(df, jd_spec)
    ranked = scorer.compute_hybrid_score(df)

    if use_llm_rerank:
        print(f"\n[6/6] LLM re-ranking top {top_n} candidates...")
        top_df = ranked.head(top_n)
        top_df = llm_rerank(top_df, jd_text)
        top_df = top_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        ranked = pd.concat([top_df, ranked.iloc[top_n:]], ignore_index=True)
    else:
        print("\n[6/6] Skipping LLM re-rank (add --llm-rerank to enable)")

    print(f"\n  Building submission CSV (top {top_n})...")
    submission = scorer.build_submission(ranked, top_n=top_n)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"\n{'='*55}")
    print(f"  Done! -> {output_path}")
    print(f"{'='*55}")
    print(submission.head(10).to_string(index=False))
    print(f"\nValidate with:  python validate_submission.py {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output",
                        default=os.path.join(config.OUTPUT_DIR, "ranked_candidates.csv"))
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--llm-jd", action="store_true")
    parser.add_argument("--llm-rerank", action="store_true")
    parser.add_argument("--use-transformers", action="store_true",
                        help="Use sentence-transformers instead of TF-IDF (slower but more semantic)")
    args = parser.parse_args()

    run_pipeline(
        jd_path=args.jd,
        candidates_path=args.candidates,
        output_path=args.output,
        top_n=args.top,
        use_llm_jd=args.llm_jd,
        use_llm_rerank=args.llm_rerank,
        use_transformers=args.use_transformers,
    )


if __name__ == "__main__":
    main()
