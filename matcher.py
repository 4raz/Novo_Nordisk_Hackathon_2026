"""
matcher.py — Vendor matching against a sampled master database.

Architecture (100 K-row subset)
-------------------------------
1.  **Load & sample** — read only ``nrows`` rows from the CSV.
2.  **Pre-clean** — vectorised cleaning of name + domain columns at init.
3.  **Domain index** — ``dict[str, list[int]]`` for O(1) exact-domain lookup.
4.  **Query pipeline** per incoming payload:
        Stage 1 — Collect candidates via domain-index hit  *and*
                  ``rapidfuzz.process.extract`` on clean names (top 200).
        Stage 2 — Score every candidate with a composite formula
                  (token-sort 40 % + partial-ratio 20 % + domain 40 %).
        Stage 3 — Return top K, sorted by composite score.
"""

from __future__ import annotations

import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz, process

from cleaner import clean_company_name, extract_root_domain, normalize_country, clean_record

# ---------------------------------------------------------------------------
# CSV column mapping  (actual columns in novo_vendor_master.csv)
# ---------------------------------------------------------------------------
_CSV_FIELD_MAP = {
    "name": "name",
    "website": "website",
    "country": "country_code",
}


class AlgorithmicMatcher:
    """Fast, indexed vendor matcher for a sampled master database."""

    def __init__(
        self,
        master_db_path: str = "novo_vendor_master.csv",
        nrows: int = 100_000,
    ):
        print(f"[matcher] Loading first {nrows:,} rows from {master_db_path} …")
        self.master_df = pd.read_csv(
            master_db_path,
            nrows=nrows,
            dtype=str,                
            keep_default_na=False,      
        )
        print("[matcher] Cleaning names & domains …")
        self.master_df["clean_name"] = (
            self.master_df["name"].apply(clean_company_name)
        )
        self.master_df["clean_domain"] = (
            self.master_df["website"].apply(extract_root_domain)
        )
        self.master_df["clean_country"] = (
            self.master_df["country_code"].apply(normalize_country)
        )

        print("[matcher] Building domain index …")
        self._domain_index: dict[str, list[int]] = defaultdict(list)
        for idx, domain in enumerate(self.master_df["clean_domain"]):
            if domain:
                self._domain_index[domain].append(idx)

        self._name_list: list[str] = self.master_df["clean_name"].tolist()

        print(f"[matcher] Ready — {len(self.master_df):,} vendors indexed.\n")

    def find_candidates(
        self,
        input_payload: dict,
        top_k: int = 5,
        name_candidates: int = 200,
    ) -> list[dict]:
        """Return the *top_k* best-matching vendors for *input_payload*.

        Parameters

        input_payload : dict
            Must contain at least ``name``; optionally ``website`` and ``country``.
        top_k : int
            How many results to return.
        name_candidates : int
            How many rough candidates to pull from the name-similarity stage.
        """
        cleaned = clean_record(input_payload)
        target_name = cleaned["clean_name"]
        target_domain = cleaned["clean_domain"]
        target_country = cleaned["country"]

        if not target_name and not target_domain:
            return []

        domain_hits: set[int] = set()
        if target_domain and target_domain in self._domain_index:
            domain_hits = set(self._domain_index[target_domain])

        name_hits: set[int] = set()
        if target_name:
            # process.extract returns list of (match, score, index)
            fuzzy_results = process.extract(
                target_name,
                self._name_list,
                scorer=fuzz.token_sort_ratio,
                limit=name_candidates,
                score_cutoff=40,        # ignore very weak matches
            )
            name_hits = {idx for _, _, idx in fuzzy_results}

        # Union of candidates from both stages
        candidate_indices = domain_hits | name_hits
        if not candidate_indices:
            return []

        # Stage 2: Detailed scoring
        results: list[dict] = []
        for idx in candidate_indices:
            row = self.master_df.iloc[idx]
            row_name = row["clean_name"]
            row_domain = row["clean_domain"]
            row_country = row["clean_country"]

            # Name score  (token-sort 65 % + partial-ratio 35 %)
            token_sort = fuzz.token_sort_ratio(target_name, row_name) if target_name else 0.0
            partial = fuzz.partial_ratio(target_name, row_name) if target_name else 0.0
            name_score = token_sort * 0.65 + partial * 0.35

            # Domain score
            if target_domain and row_domain:
                domain_score = 100.0 if target_domain == row_domain else fuzz.ratio(target_domain, row_domain)
            else:
                domain_score = 0.0

            # Country bonus / penalty
            country_bonus = 0.0
            if target_country and row_country:
                country_bonus = 5.0 if target_country == row_country else -2.0

            # Composite  (55 % name, 40 % domain, 5 % country)
            composite = (name_score * 0.55) + (domain_score * 0.40) + country_bonus

            results.append({
                "handle": row["handle"],
                "name": row["name"],
                "website": row["website"],
                "country_code": row["country_code"],
                "industry": row.get("industry", ""),
                "city": row.get("city", ""),
                "composite_score": round(composite, 2),
                "name_score": round(name_score, 2),
                "domain_score": round(domain_score, 2),
            })

        # ── Sort and return top K ────────────────────────────────────────
        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results[:top_k]