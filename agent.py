"""
agent.py — Agentic validation layer for the Vendor Name Resolution pipeline.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any
import requests
from openai import AzureOpenAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

VALID_DECISIONS = {"AUTO_REJECT", "MANUAL_REVIEW", "AUTO_APPROVE"}

OPENSANCTIONS_MATCH_URL = "https://api.opensanctions.org/match/default"
GLEIF_API_BASE = "https://api.gleif.org/api/v1"


def query_opensanctions(vendor_name: str) -> dict:
    if not vendor_name:
        return {"status": "error", "sanctions_hit": None, "hits": []}

    api_key = os.environ.get("OPENSANCTIONS_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    payload = {
        "queries": {"q1": {"schema": "Company", "properties": {"name": [vendor_name]}}}
    }

    try:
        response = requests.post(
            OPENSANCTIONS_MATCH_URL, json=payload, headers=headers, timeout=8
        )
        response.raise_for_status()
        results = response.json().get("responses", {}).get("q1", {}).get("results", [])
        flagged = [
            {"caption": r.get("caption"), "match_score": r.get("score", 0)}
            for r in results
            if r.get("score", 0) >= 0.70
        ]
        return {"status": "success", "sanctions_hit": bool(flagged), "hits": flagged}
    except requests.exceptions.RequestException as exc:
        return {
            "status": "error",
            "sanctions_hit": None,
            "hits": [],
            "note": f"OpenSanctions lookup failed: {exc}",
        }


def query_gleif_hierarchy(vendor_name: str) -> dict:
    if not vendor_name:
        return {"status": "error", "hierarchy_found": None, "matched_entity": None}

    try:
        search_url = f"{GLEIF_API_BASE}/fuzzycompletions"
        response = requests.get(
            search_url,
            params={"field": "entity.legalName", "q": vendor_name},
            timeout=8,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        if not data:
            return {
                "status": "no_record_found",
                "hierarchy_found": False,
                "matched_entity": None,
            }

        lei = (
            data[0]
            .get("relationships", {})
            .get("lei-records", {})
            .get("data", {})
            .get("id")
        )
        if not lei:
            return {
                "status": "no_record_found",
                "hierarchy_found": False,
                "matched_entity": None,
            }

        parent_url = f"{GLEIF_API_BASE}/lei-records/{lei}/ultimate-parent"
        parent_response = requests.get(parent_url, timeout=8)
        if parent_response.status_code == 200:
            parent_data = parent_response.json().get("data")
            if parent_data:
                parent_name = (
                    parent_data.get("attributes", {})
                    .get("entity", {})
                    .get("legalName", {})
                    .get("name")
                )
                if parent_name:
                    return {
                        "status": "hierarchy_found",
                        "hierarchy_found": True,
                        "matched_entity": parent_name,
                    }

        direct_url = f"{GLEIF_API_BASE}/lei-records/{lei}/direct-parent"
        direct_response = requests.get(direct_url, timeout=8)
        if direct_response.status_code == 200:
            direct_data = direct_response.json().get("data")
            if direct_data:
                parent_name = (
                    direct_data.get("attributes", {})
                    .get("entity", {})
                    .get("legalName", {})
                    .get("name")
                )
                if parent_name:
                    return {
                        "status": "hierarchy_found",
                        "hierarchy_found": True,
                        "matched_entity": parent_name,
                    }

        return {
            "status": "no_hierarchy_risk",
            "hierarchy_found": False,
            "matched_entity": None,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "status": "error",
            "hierarchy_found": None,
            "matched_entity": None,
            "note": f"GLEIF lookup failed: {exc}",
        }


AZURE_OPENAI_DEPLOYMENT_NAME = (
    os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    or "gpt-4o-mini"
)


def _get_azure_client() -> "AzureOpenAI | None":
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get(
        "AZURE_OPENAI_KEY"
    )
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        return None
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    try:
        return AzureOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version
        )
    except Exception:
        return None


_SYSTEM_PROMPT = """You are an enterprise procurement compliance analyst for Novo Nordisk. 
Evaluate new vendor onboarding requests against an internal vendor-master match and external risk registries.

Rules for interpreting the external data:
1. Watchlist screening: a common or generic name can trigger a false positive. Only set found_sanction_risk to true if the data genuinely points to THIS vendor.
2. Corporate hierarchy: Only flag found_hierarchy_risk as true if the GLEIF data explicitly identifies a corporate parent or subsidiary relationship.
3. Master Data Duplicates: Use your contextual judgment to determine if the internal database match represents a true duplicate, a regional franchise, or a false positive based on the name, domain, and country.

Output requirement:
Write business_justification in plain English for a non-technical data steward, as short markdown bullet points. Never mention technical metrics, percentages, or algorithmic weights.

Respond with a single JSON object only with exactly these keys:
  recommended_decision: "AUTO_APPROVE" | "MANUAL_REVIEW" | "AUTO_REJECT"
  business_justification: string, markdown bullet points
  found_sanction_risk: true | false
  found_hierarchy_risk: true | false
  related_parent_entity: string or null
"""


def _build_user_prompt(
    vendor_payload: dict,
    match_result: "dict | None",
    composite_score: float,
    sanctions_data: dict,
    gleif_data: dict,
) -> str:
    return (
        f"Proposed vendor: {vendor_payload.get('name', '')} ({vendor_payload.get('country', '')})\n"
        f"Website: {vendor_payload.get('website', '')}\n"
        f"Top internal master-data match: "
        f"{match_result.get('name', 'None') if match_result else 'None'} "
        f"(ID: {match_result.get('vendor_id', 'None') if match_result else 'None'})\n"
        f"Internal composite similarity score: {composite_score:.1f}\n\n"
        f"OpenSanctions watchlist data (JSON): {json.dumps(sanctions_data)}\n"
        f"GLEIF corporate-registry data (JSON): {json.dumps(gleif_data)}"
    )


def _call_llm(
    client: AzureOpenAI,
    vendor_payload: dict,
    match_result: "dict | None",
    composite_score: float,
    sanctions_data: dict,
    gleif_data: dict,
) -> "dict | None":
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        vendor_payload,
                        match_result,
                        composite_score,
                        sanctions_data,
                        gleif_data,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return _safe_parse_json(response.choices[0].message.content)
    except Exception:
        return None


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _safe_parse_json(text: "str | None") -> "dict | None":
    if not text or not isinstance(text, str):
        return None
    candidates = [text.strip()]
    if fence_match := _JSON_FENCE_RE.search(text):
        candidates.append(fence_match.group(1).strip())
    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _merge_risk_signal(
    raw_hit: "bool | None", llm_says_risk: "bool | None"
) -> "bool | None":
    if raw_hit is None:
        return None
    if raw_hit is False:
        return False
    return bool(llm_says_risk) if llm_says_risk is not None else True


def _apply_routing_rules(
    composite_score: float,
    sanctions_hit: "bool | None",
    hierarchy_found: "bool | None",
    related_parent: "str | None",
    llm_justification: str,
    llm_recommended_decision: "str | None" = None,
) -> tuple[str, str, list[str]]:

    if sanctions_hit:
        decision = "AUTO_REJECT"
        notes = [
            "Priority 1: confirmed sanctions/watchlist hit -- blocks the vendor regardless of any other signal."
        ]
        justification = f"**🛑 CRITICAL COMPLIANCE BLOCK:** A sanctions or watchlist risk was identified for this vendor; onboarding is not permitted.\n\n{llm_justification}"

    elif hierarchy_found:
        decision = "MANUAL_REVIEW"
        parent_text = related_parent if related_parent else "a corporate parent"
        notes = [
            f"Priority 2: a parent-subsidiary relationship was identified to {parent_text}. Overrides duplicate check."
        ]
        justification = f"**⚠️ HIERARCHY IDENTIFIED:** Global records indicate this entity is linked to **{parent_text}**. Route to a data steward to determine whether its parent already exists (in which case we consolidate spending) or to start a new entry under its parent's name.\n\n{llm_justification}"

    elif composite_score >= 95.0:
        if llm_recommended_decision == "MANUAL_REVIEW":
            decision = "MANUAL_REVIEW"
            notes = [
                f"Priority 3 (Downgraded): Score {composite_score:.1f}% is a duplicate, but AI contextually requested MANUAL_REVIEW."
            ]
            justification = f"**⚠️ REVIEW REQUIRED:** This request is a near-identical match to an existing record, but requires contextual human verification.\n\n{llm_justification}"
        else:
            decision = "AUTO_REJECT"
            notes = [
                f"Priority 3: internal similarity score {composite_score:.1f}% is >= 95%, treated as an exact master-data duplicate."
            ]
            justification = f"**🛑 DUPLICATE BLOCKED:** This request is a near-identical match to an existing vendor-master record.\n\n{llm_justification}"

    elif composite_score >= 85.0:
        decision = "MANUAL_REVIEW"
        notes = [
            f"Priority 4: internal similarity score {composite_score:.1f}% is in the 85-94.9% ambiguity band."
        ]
        justification = f"**⚠️ REVIEW REQUIRED:** This proposed vendor shares significant overlap with an existing record; manual verification is needed before proceeding.\n\n{llm_justification}"

    else:
        if llm_recommended_decision == "MANUAL_REVIEW":
            decision = "MANUAL_REVIEW"
            notes = [
                f"Priority 5 (Upgraded): Score {composite_score:.1f}% cleared deterministic rules, but AI detected nuanced risk and requested MANUAL_REVIEW."
            ]
            justification = f"**⚠️ REVIEW REQUIRED:** The automated AI agent flagged contextual ambiguities requiring steward review.\n\n{llm_justification}"
        else:
            decision = "AUTO_APPROVE"
            notes = [
                "Priority 5: no sanctions, hierarchy, or duplicate-score risk detected -- cleared."
            ]
            justification = f"**✅ APPROVED:** No duplicates, sanctions, or corporate-hierarchy conflicts were detected for this vendor.\n\n{llm_justification}"

    if llm_recommended_decision in VALID_DECISIONS:
        if llm_recommended_decision != decision:
            notes.append(
                f"Reasoning engine suggested {llm_recommended_decision}; deterministic rules enforced {decision} instead."
            )
        else:
            notes.append(
                f"Reasoning engine's suggested decision ({llm_recommended_decision}) matches the deterministic outcome."
            )

    return decision, justification.strip(), notes


def run_validation(vendor_payload: dict, match_result: "dict | None" = None) -> dict:
    try:
        composite_score = (
            float(match_result.get("composite_score", 0.0)) if match_result else 0.0
        )
    except (TypeError, ValueError):
        composite_score = 0.0

    target_name = (vendor_payload or {}).get("name", "") or ""

    sanctions_data = query_opensanctions(target_name)
    gleif_data = query_gleif_hierarchy(target_name)

    client = _get_azure_client()

    if client is None:
        sanctions_hit = sanctions_data.get("sanctions_hit")
        hierarchy_found = gleif_data.get("hierarchy_found")
        related_parent = gleif_data.get("matched_entity") if hierarchy_found else None
        llm_text = "- Automated reasoning engine is offline; this result reflects deterministic rule-based screening only."
        decision, justification, notes = _apply_routing_rules(
            composite_score, sanctions_hit, hierarchy_found, related_parent, llm_text
        )
        notes.append(
            "Azure OpenAI credentials not configured -- ran in deterministic offline mode."
        )
        return {
            "decision": decision,
            "business_justification": justification,
            "similarity_score": round(composite_score, 2),
            "sanctions_hit": sanctions_hit,
            "hierarchy_relationship_found": hierarchy_found,
            "related_parent_entity": related_parent,
            "safety_net_notes": notes,
        }

    llm_data = _call_llm(
        client,
        vendor_payload,
        match_result,
        composite_score,
        sanctions_data,
        gleif_data,
    )

    if llm_data is None:
        if sanctions_data.get("sanctions_hit") is True:
            decision = "AUTO_REJECT"
            justification = "**🛑 CRITICAL COMPLIANCE BLOCK:** A sanctions or watchlist risk was identified for this vendor; onboarding is not permitted.\n\n- The automated reasoning engine was unavailable; this block is based on the compliance watchlist check alone."
            notes = [
                "Priority 1: confirmed sanctions/watchlist hit -- blocks the vendor regardless of the reasoning-engine failure."
            ]
        else:
            decision = "MANUAL_REVIEW"
            justification = "**⚠️ REVIEW REQUIRED:** The automated reasoning engine could not complete this evaluation, so this request is routed to a data steward as a precaution.\n\n- Please verify manually against the internal match and any compliance/hierarchy signals."
            notes = [
                "Automated reasoning engine failed or returned an unparseable response; routed to manual review as a precaution."
            ]
        return {
            "decision": decision,
            "business_justification": justification,
            "similarity_score": round(composite_score, 2),
            "sanctions_hit": sanctions_data.get("sanctions_hit"),
            "hierarchy_relationship_found": gleif_data.get("hierarchy_found"),
            "related_parent_entity": (
                gleif_data.get("matched_entity")
                if gleif_data.get("hierarchy_found")
                else None
            ),
            "safety_net_notes": notes,
        }

    sanctions_hit = _merge_risk_signal(
        sanctions_data.get("sanctions_hit"), llm_data.get("found_sanction_risk")
    )
    hierarchy_found = _merge_risk_signal(
        gleif_data.get("hierarchy_found"), llm_data.get("found_hierarchy_risk")
    )
    related_parent = llm_data.get("related_parent_entity") or (
        gleif_data.get("matched_entity") if hierarchy_found else None
    )
    llm_justification = (
        llm_data.get("business_justification")
        or "- No further detail was provided by the reasoning engine."
    )

    decision, justification, notes = _apply_routing_rules(
        composite_score,
        sanctions_hit,
        hierarchy_found,
        related_parent,
        llm_justification,
        llm_recommended_decision=llm_data.get("recommended_decision"),
    )

    return {
        "decision": decision,
        "business_justification": justification,
        "similarity_score": round(composite_score, 2),
        "sanctions_hit": sanctions_hit,
        "hierarchy_relationship_found": hierarchy_found,
        "related_parent_entity": related_parent,
        "safety_net_notes": notes,
    }
