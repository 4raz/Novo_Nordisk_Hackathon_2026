"""
matcher.py — Intelligent Vendor Matching via Fuzzy Algorithms and Semantic Vectors.
"""

from __future__ import annotations
import os

import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz, process
import torch
from sentence_transformers import SentenceTransformer, util

# Only importing clean_record to clean incoming search payloads
from cleaner import clean_record

_CSV_FIELD_MAP = {
    "name": "name",
    "website": "website",
    "country": "country_code",
}


class AlgorithmicMatcher:
    """Hybrid vendor matcher combining RapidFuzz string metrics with NLP embeddings."""

    def __init__(
        self,
        master_db_path: str = "novo_vendor_master.csv",
        nrows: int = 100_000,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_path: str = "vendor_embeddings.pt",
        cleaned_csv_path: str = "novo_vendor_master_cleaned.csv",
    ):
        import os

        # 1. Load DataFrame & Embeddings
        if os.path.exists(embeddings_path) and os.path.exists(cleaned_csv_path):
            print("[matcher] Loading precomputed database and vector tensor...")
            self.master_df = pd.read_csv(
                cleaned_csv_path,
                nrows=nrows,
                dtype=str,
                keep_default_na=False,
            )
            self.name_embeddings = torch.load(embeddings_path, weights_only=False)
            print(f"[matcher] Initializing Semantic AI Model ({model_name}) …")
            self.model = SentenceTransformer(model_name)
        else:
            print(f"[matcher] Loading master database ({nrows:,} rows) …")
            self.master_df = pd.read_csv(
                master_db_path,
                nrows=nrows,
                dtype=str,
                keep_default_na=False,
            )
            print(f"[matcher] Initializing Semantic AI Model ({model_name}) …")
            self.model = SentenceTransformer(model_name)
            name_list = self.master_df["clean_name"].tolist()
            self.name_embeddings = self.model.encode(name_list, convert_to_tensor=True)

        # 2. Build In-Memory Lookups (Required for Candidate Retrieval)
        print("[matcher] Building O(1) domain index …")
        self._domain_index: dict[str, list[int]] = defaultdict(list)
        for idx, domain in enumerate(self.master_df["clean_domain"]):
            if domain:
                self._domain_index[domain].append(idx)

        self._name_list: list[str] = self.master_df["clean_name"].tolist()

        print(f"[matcher] System ready — {len(self.master_df):,} vendors indexed.\n")

    def find_candidates(
        self,
        input_payload: dict,
        top_k: int = 5,
        name_candidates: int = 200,
    ) -> list[dict]:
        """Return the top_k best-matching vendors using multi-signal scoring."""
        # We still clean the incoming payload so it matches the DB formatting
        cleaned = clean_record(input_payload)
        target_name = cleaned["clean_name"]
        target_domain = cleaned["clean_domain"]
        target_country = cleaned["country"]

        if not target_name and not target_domain:
            return []

        # 1. Fast Candidate Blocking
        domain_hits: set[int] = set()
        if target_domain and target_domain in self._domain_index:
            domain_hits = set(self._domain_index[target_domain])

        name_hits: set[int] = set()
        if target_name:
            # fuzz.WRatio handles case, length differences, and substring matches better than token_sort_ratio
            fuzzy_results = process.extract(
                target_name,
                self._name_list,
                scorer=fuzz.WRatio,
                limit=name_candidates,
                score_cutoff=50,
            )
            name_hits = {idx for _, _, idx in fuzzy_results}

        candidate_indices = list(domain_hits | name_hits)
        if not candidate_indices:
            return []

        # 2. Compute Target Vector
        target_embedding = (
            self.model.encode([target_name], convert_to_tensor=True)
            if target_name
            else None
        )

        # 3. Deep Scoring Execution
        results: list[dict] = []
        for idx in candidate_indices:
            row = self.master_df.iloc[idx]
            row_name = row["clean_name"]
            row_domain = row["clean_domain"]
            row_country = row["clean_country"]

            # A. Advanced String Similarities
            if target_name and row_name:
                # token_set_ratio perfectly aligns strings with extra trailing/leading words
                set_score = fuzz.token_set_ratio(target_name, row_name)
                # WRatio anchors formatting shifts and general character alignments
                w_score = fuzz.WRatio(target_name, row_name)
                algorithmic_name_score = (set_score * 0.6) + (w_score * 0.4)

                # B. Semantic Vector Similarity
                row_embedding = self.name_embeddings[idx].unsqueeze(0)
                semantic_score = (
                    util.cos_sim(target_embedding, row_embedding).item() * 100.0
                )

                # Use semantic score if it detects an acronym/alias the string logic missed
                final_name_score = max(algorithmic_name_score, semantic_score)
            else:
                final_name_score = 0.0
                algorithmic_name_score = 0.0
                semantic_score = 0.0

            # C. Domain Scoring
            if target_domain and row_domain:
                if target_domain == row_domain:
                    domain_score = 100.0
                else:
                    # Strip the TLD (.com, .net) to prevent artificial overlap
                    target_base = target_domain.split(".")[0]
                    row_base = row_domain.split(".")[0]

                    # Apply a severe multiplier penalty; mismatched domains strongly suggest distinct entities
                    domain_score = fuzz.ratio(target_base, row_base) * 0.5
            else:
                domain_score = 0.0

            # D. Proportional Country Penalty
            country_multiplier = 1.0
            if target_country and row_country:
                if target_country != row_country:
                    country_multiplier = 0.85  # 15% penalty for differing countries

            # E. Final Composite Score
            base_score = (final_name_score * 0.55) + (domain_score * 0.45)
            composite = base_score * country_multiplier

            results.append(
                {
                    "vendor_id": row.get("handle", f"VN-{idx}"),
                    "name": row["name"],
                    "website": row.get("website", ""),
                    "country_code": row.get("country_code", ""),
                    "industry": row.get("industry", ""),
                    "city": row.get("city", ""),
                    "composite_score": round(composite, 2),
                    "name_score": round(final_name_score, 2),
                    "semantic_score": round(semantic_score, 2),
                    "fuzzy_score": round(algorithmic_name_score, 2),
                    "domain_score": round(domain_score, 2),
                }
            )

        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results[:top_k]
