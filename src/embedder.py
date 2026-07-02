"""
Semantic similarity module.

Default (--fast): TF-IDF + cosine similarity. Instant, no downloads.
Optional (--slow): sentence-transformers all-MiniLM-L6-v2 (smaller/faster model).

For a 5000-candidate dataset on a normal laptop, TF-IDF completes in seconds
and still captures meaningful vocabulary overlap. The full sentence-transformer
model is overkill for most hackathon-scale datasets.
"""

# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _tfidf_similarity(jd_text: str, candidate_texts: list) -> np.ndarray:
    """Fast TF-IDF cosine similarity — no model download needed."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),   # bigrams help catch "machine learning", "data engineering" etc.
        sublinear_tf=True,    # dampens very frequent terms
    )
    corpus = [jd_text] + candidate_texts
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    return np.clip(sims, 0, 1)


def _transformer_similarity(jd_text: str, candidate_texts: list) -> np.ndarray:
    """Slower but more semantic — only used if --use-transformers flag is set."""
    global _model
    if _model is None:
        # pyrefly: ignore [missing-import]
        from sentence_transformers import SentenceTransformer
        print("      Loading all-MiniLM-L6-v2 (small fast model ~80MB)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    jd_vec = _model.encode([jd_text], convert_to_numpy=True, show_progress_bar=False)
    cand_vecs = _model.encode(
        candidate_texts, convert_to_numpy=True,
        show_progress_bar=True, batch_size=128
    )
    sims = cosine_similarity(jd_vec, cand_vecs)[0]
    return (np.clip(sims, -1, 1) + 1) / 2


def semantic_similarity(jd_text: str, candidate_texts: list,
                        use_transformers: bool = False) -> np.ndarray:
    if use_transformers:
        return _transformer_similarity(jd_text, candidate_texts)
    print("      Using TF-IDF similarity (fast mode)...")
    return _tfidf_similarity(jd_text, candidate_texts)
