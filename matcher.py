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
        pool_size: int = 500,
    ) -> list[dict]:
        """Parallel Sandbox: Retrieves top candidates via AI and Fuzzy simultaneously."""
        cleaned = clean_record(input_payload)
        target_name = cleaned["clean_name"]
        target_domain = cleaned["clean_domain"]
        target_country = cleaned["country"]

        if not target_name and not target_domain:
            return []

        # --- STAGE 1: PARALLEL GLOBAL RETRIEVAL ---
        candidate_indices = set()

        # 1A. O(1) Exact Domain Hits
        if target_domain and target_domain in self._domain_index:
            candidate_indices.update(self._domain_index[target_domain])

        # 1B. Global Semantic AI Search
        target_embedding = None
        cos_scores = None
        if target_name:
            target_embedding = self.model.encode([target_name], convert_to_tensor=True)
            cos_scores = util.cos_sim(target_embedding, self.name_embeddings)[0]

            top_semantic = cos_scores.topk(
                k=min(pool_size, len(self.master_df))
            ).indices.tolist()
            candidate_indices.update(top_semantic)

        # 1C. Global Lexical Search
        if target_name:
            fuzzy_results = process.extract(
                target_name,
                self._name_list,
                scorer=fuzz.WRatio,
                limit=pool_size,
                score_cutoff=40,
            )
            top_fuzzy = [idx for _, _, idx in fuzzy_results]
            candidate_indices.update(top_fuzzy)

        if not candidate_indices:
            return []

        # --- STAGE 2: DEEP COMPOSITE SCORING ON THE SANDBOX ---
        results: list[dict] = []
        for idx in candidate_indices:
            row = self.master_df.iloc[idx]
            row_name = row["clean_name"]
            row_domain = row["clean_domain"]
            row_country = row["clean_country"]

            # A. Advanced String Similarities
            if target_name and row_name:
                set_score = fuzz.token_set_ratio(target_name, row_name)
                w_score = fuzz.WRatio(target_name, row_name)

                # Base algorithmic score favors subsets (allows "XYZ" to match "XYZ Industries")
                base_alg = (set_score * 0.65) + (w_score * 0.35)

                # NEW: Compound Word Fix (Removes spaces to match "Bluetech" with "Blue Tech")
                spaceless_score = fuzz.ratio(
                    target_name.replace(" ", ""), row_name.replace(" ", "")
                )

                # Take the highest score between the subset logic and the spaceless logic
                base_alg = max(base_alg, spaceless_score)

                # Token Coverage Penalty to prevent single-word fragments from winning
                target_tokens = set(target_name.lower().split())
                row_tokens = set(row_name.lower().split())

                if len(target_tokens) > 1 and len(row_tokens) == 1:
                    base_alg *= 0.88

                algorithmic_name_score = base_alg

                # Fetch semantic score using the standard target vector
                if target_embedding is not None and cos_scores is not None:
                    semantic_score = cos_scores[idx].item() * 100.0
                else:
                    semantic_score = 0.0

                final_name_score = max(algorithmic_name_score, semantic_score)

            # B. Domain Scoring & Dynamic Weighting
            if target_domain and row_domain:
                if target_domain == row_domain:
                    domain_score = 100.0
                    # Golden Signal: If domains match exactly, guarantee a high composite score
                    # to override acronym/name discrepancies (e.g. IBM vs International Business Machines)
                    base_score = max((final_name_score * 0.30) + 70.0, 96.0)
                else:
                    target_base = target_domain.split(".")[0]
                    row_base = row_domain.split(".")[0]
                    domain_score = fuzz.ratio(target_base, row_base) * 0.5
                    base_score = (final_name_score * 0.55) + (domain_score * 0.45)
            else:
                domain_score = 0.0
                base_score = final_name_score

            # C. Proportional Country Penalty & Global Tenant Exemption
            country_multiplier = 1.0

            # EXEMPTION: If they share an exact domain, they are a verified global subsidiary
            # (e.g., IBM India & IBM US). Waive all geographic penalties.
            if target_domain and row_domain and target_domain == row_domain:
                country_multiplier = 1.0

            # PENALTY: Penalize geographic mismatches for vendors without shared domains
            elif target_country and row_country:
                if target_country != row_country:
                    country_multiplier = 0.85

            # PENALTY: If one record is missing a country entirely, apply a minor unverified penalty
            # so sparse records (like alpha-beta) don't unfairly outscore fully populated ones.
            elif bool(target_country) != bool(row_country):
                country_multiplier = 0.95

            # D. Final Composite Calculation
            composite = base_score * country_multiplier

            # Silent Tie-Breaker: Bypasses the cleaner to see which raw string matches best
            raw_input = str(input_payload.get("name", "")).lower()
            raw_row = str(row["name"]).lower()
            raw_tiebreaker = fuzz.ratio(raw_input, raw_row)

            results.append(
                {
                    "vendor_id": row.get("vendor_id", f"VN-{idx}"),
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
                    "raw_tiebreaker": raw_tiebreaker,  # Added strictly for sorting ties
                }
            )

        # Sort by Composite Score first. If tied, the raw string ratio breaks the tie.
        results.sort(
            key=lambda r: (r["composite_score"], r["raw_tiebreaker"]), reverse=True
        )
        return results[:top_k]
