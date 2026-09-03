"""
agent.py — Agentic validation layer for the Vendor Name Resolution pipeline.

Where this sits in the pipeline
--------------------------------
    clean (cleaner.py) -> find & score (matcher.AlgorithmicMatcher) -> VALIDATE (here) -> recommend

What this module provides
--------------------------
1. Two LangChain `@tool` functions backed by live, public data sources:
     - check_corporate_hierarchy   -> GLEIF Global LEI Index, Level 2 (parent/
       subsidiary) relationship data. Free, no API key required.
     - check_un_sanctions -> UN Security Council & UNGM Ineligible Vendor
       screening via the OpenSanctions API.
2. An AzureChatOpenAI reasoning engine at temperature=0.1, bound to
   AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT_NAME.
3. A strict system prompt encoding the enterprise routing rules (AUTO_REJECT /
   MANUAL_REVIEW / AUTO_APPROVE, in priority order) and a JSON-only output
   contract with a two-sentence business justification.
4. create_tool_calling_agent + AgentExecutor wiring, plus run_validation(): a
   safe wrapper that parses the agent's JSON, falls back to a MANUAL_REVIEW
   dict on any formatting or execution failure, cross-checks the agent's
   decision against the raw tool evidence (never the LLM's restatement of it),
   and re-injects the deterministic algorithmic score for the Streamlit UI.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

# --------------------------------------------------------------------------- #
# 1. External API Query Tools
# --------------------------------------------------------------------------- #

GLEIF_API_BASE = "https://api.gleif.org/api/v1"
OPENSANCTIONS_API_BASE = "https://api.opensanctions.org/search/default"


def _gleif_get(
    path: str, params: dict | None = None, timeout: int = 10
) -> requests.Response:
    return requests.get(f"{GLEIF_API_BASE}{path}", params=params, timeout=timeout)


def _lookup_lei(vendor_legal_name: str) -> dict | None:
    for filter_field in ("entity.legalName", "fulltext"):
        resp = _gleif_get(
            "/lei-records",
            params={f"filter[{filter_field}]": vendor_legal_name, "page[size]": 3},
        )
        if resp.status_code != 200:
            continue
        records = resp.json().get("data", [])
        if records:
            return records[0]
    return None


def _fetch_parent(lei: str, relation: str) -> dict | None:
    resp = _gleif_get(f"/lei-records/{lei}/{relation}")
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json().get("data")
    if not data:
        return None
    entity = data.get("attributes", {}).get("entity", {})
    return {
        "lei": data.get("id"),
        "legal_name": entity.get("legalName", {}).get("name"),
        "country": entity.get("legalAddress", {}).get("country"),
    }


@tool
def check_corporate_hierarchy(vendor_legal_name: str) -> str:
    """Look up a vendor's corporate parent/subsidiary structure via the GLEIF
    Global LEI Index Level 2 (relationship) data. Pass the
    vendor's full, ORIGINAL legal name (not a cleaned/abbreviated form).
    """
    try:
        record = _lookup_lei(vendor_legal_name)
        if record is None:
            return json.dumps(
                {
                    "lei_found": False,
                    "hierarchy_relationship_found": False,
                    "note": f"No GLEIF LEI record located for '{vendor_legal_name}'.",
                }
            )

        lei = record["id"]
        matched_name = (
            record.get("attributes", {})
            .get("entity", {})
            .get("legalName", {})
            .get("name")
            or vendor_legal_name
        )

        direct_parent = _fetch_parent(lei, "direct-parent")
        ultimate_parent = _fetch_parent(lei, "ultimate-parent")

        return json.dumps(
            {
                "lei_found": True,
                "lei": lei,
                "matched_legal_name": matched_name,
                "hierarchy_relationship_found": bool(direct_parent or ultimate_parent),
                "direct_parent": direct_parent,
                "ultimate_parent": ultimate_parent,
                "source": "GLEIF Global LEI Index",
            }
        )
    except requests.exceptions.RequestException as exc:
        return json.dumps(
            {
                "lei_found": False,
                "hierarchy_relationship_found": False,
                "error": f"GLEIF lookup failed: {exc}",
            }
        )


@tool
def check_un_sanctions(vendor_legal_name: str, country_code: str = "") -> str:
    """Screen a vendor against United Nations Security Council Sanctions and
    UNGM Ineligible Vendor lists. Pass the vendor's full, ORIGINAL legal name.
    """
    api_key = os.environ.get("OPENSANCTIONS_API_KEY", "").strip()
    if not api_key:
        return json.dumps(
            {
                "screening_completed": False,
                "sanctions_hit": None,
                "note": "OPENSANCTIONS_API_KEY is not configured in .env. Unverified result.",
            }
        )

    try:
        headers = {"Authorization": f"ApiKey {api_key}"}

        # Follow the OpenSanctions Match API structure (POST request)
        request_payload = {
            "queries": {
                "vendor_query": {
                    "schema": "Organization",
                    "properties": {"name": [vendor_legal_name]},
                }
            }
        }

        resp = requests.post(
            "https://api.opensanctions.org/match/default",
            headers=headers,
            json=request_payload,
            timeout=10,
        )
        resp.raise_for_status()

        # Parse the nested Match API response
        responses = resp.json().get("responses", {})
        query_results = responses.get("vendor_query", {}).get("results", [])

        un_matches = []
        for r in query_results:
            datasets = r.get("datasets", [])
            # Filter specifically for UN/UNGM datasets
            if any("un_" in ds or "ungm" in ds for ds in datasets):
                un_matches.append(
                    {
                        "name": r.get("caption"),
                        "datasets": datasets,
                        "score": r.get("score"),
                    }
                )

        return json.dumps(
            {
                "screening_completed": True,
                "sanctions_hit": len(un_matches) > 0,
                "hit_count": len(un_matches),
                "matches": un_matches,
                "source": "OpenSanctions Match API (UN Security Council / UNGM datasets)",
            }
        )
    except requests.exceptions.RequestException as exc:
        return json.dumps(
            {
                "screening_completed": False,
                "sanctions_hit": None,
                "note": f"UN sanctions lookup failed ({exc}). Unverified result.",
            }
        )


TOOLS = [check_corporate_hierarchy, check_un_sanctions]


# --------------------------------------------------------------------------- #
# 2. Azure OpenAI LLM Connection
# --------------------------------------------------------------------------- #


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable '{name}'.")
    return value


AZURE_OPENAI_ENDPOINT = _require_env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _require_env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"
)
AZURE_OPENAI_API_VERSION = os.environ.get(
    "AZURE_OPENAI_API_VERSION", "2024-05-01-preview"
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.1,
)


# --------------------------------------------------------------------------- #
# 3. System Prompt and Decision Rules
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are the Vendor Master Data Governance Agent for Novo Nordisk's \
vendor-onboarding screening pipeline. A proposed vendor name has already been cleaned \
and matched against the internal vendor master using fuzzy-string and semantic-embedding \
algorithms. Your job is to decide -- or safely route -- whether to reuse an existing \
vendor record, send the request to a human data steward, or approve a new record.

You will be given, for one proposed vendor:
  1. The proposed vendor's raw submitted details (name, website, country, etc.).
  2. The single best-matching existing vendor-master record and its internal
     algorithmic similarity score (0-100, combining name and domain similarity).

You have two tools. Call BOTH of them, exactly once each, before you decide:
  - check_corporate_hierarchy: GLEIF Level 2 parent/subsidiary relationship data.
  - check_un_sanctions: United Nations Security Council Sanctions & UNGM lists.
Always pass the proposed vendor's ORIGINAL legal name to both tools.

ROUTING RULES -- apply in exactly this priority order:
  1. AUTO_REJECT if check_un_sanctions reports a sanctions/watchlist
     hit (sanctions_hit is true). This overrides every rule below it.
  2. MANUAL_REVIEW if check_corporate_hierarchy reports a parent-subsidiary
     relationship (hierarchy_relationship_found is true) -- even if the
     similarity score is high. 
  3. AUTO_REJECT if no hierarchy relationship was found AND the internal
     similarity score is strictly greater than 85 -- a high-confidence
     duplicate of an unrelated (non-subsidiary) existing vendor.
  4. MANUAL_REVIEW if the internal similarity score is between 60 and 85
     inclusive -- the ambiguity zone.
  5. AUTO_APPROVE only if the similarity score is below 60 AND
     check_un_sanctions completed successfully with no hit. If
     compliance screening could not be completed, route to
     MANUAL_REVIEW instead.

OUTPUT FORMAT -- this is a strict constraint. Respond with ONLY a single valid
JSON object: no markdown code fences, no prose before or after it, nothing
else. The JSON object must contain exactly these keys:

  decision: one of "AUTO_REJECT", "MANUAL_REVIEW", "AUTO_APPROVE"
  business_justification: a string containing EXACTLY two sentences, written
    for a data steward. Sentence 1 states the recommended action and its
    primary driver. Sentence 2 cites the specific supporting evidence (the
    similarity score and/or the tool findings) behind it.
  sanctions_hit: true or false
  hierarchy_relationship_found: true or false
  related_parent_entity: the legal name of the parent entity if one was
    found, otherwise null
"""

# --------------------------------------------------------------------------- #
# 4. Assemble and Execute the Agentic Pipeline
# --------------------------------------------------------------------------- #

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=6,
)

VALID_DECISIONS = {"AUTO_REJECT", "MANUAL_REVIEW", "AUTO_APPROVE"}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _safe_parse_json(text: str) -> dict | None:
    if not text or not isinstance(text, str):
        return None

    candidates = [text.strip()]
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_tool_evidence(intermediate_steps: list) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "sanctions_hit": None,
        "screening_completed": None,
        "hierarchy_relationship_found": None,
        "related_parent_entity": None,
    }
    for step in intermediate_steps or []:
        try:
            _, observation = step
        except (TypeError, ValueError):
            continue
        obs = _safe_parse_json(observation) if isinstance(observation, str) else None
        if not isinstance(obs, dict):
            continue
        if "sanctions_hit" in obs:
            evidence["sanctions_hit"] = obs.get("sanctions_hit")
            evidence["screening_completed"] = obs.get("screening_completed")
        if "hierarchy_relationship_found" in obs:
            evidence["hierarchy_relationship_found"] = obs.get(
                "hierarchy_relationship_found"
            )
            parent = obs.get("direct_parent") or obs.get("ultimate_parent")
            if parent:
                evidence["related_parent_entity"] = parent.get("legal_name")
    return evidence


def _apply_safety_net(parsed: dict, composite_score: float, evidence: dict) -> dict:
    decision = parsed.get("decision")
    notes: list[str] = []

    if evidence.get("sanctions_hit") is True and decision != "AUTO_REJECT":
        notes.append(
            "Overridden to AUTO_REJECT: the compliance tool reported a watchlist "
            "hit, which is a hard rule regardless of the agent's own answer."
        )
        decision = "AUTO_REJECT"
    elif decision == "AUTO_APPROVE":
        if evidence.get("screening_completed") is False:
            notes.append(
                "Overridden to MANUAL_REVIEW: compliance watchlist screening did "
                "not complete, so a clean result cannot be confirmed."
            )
            decision = "MANUAL_REVIEW"
        elif evidence.get("hierarchy_relationship_found") is True:
            notes.append(
                "Overridden to MANUAL_REVIEW: a parent-subsidiary relationship was found."
            )
            decision = "MANUAL_REVIEW"
        elif composite_score >= 60:
            notes.append(
                f"Overridden to MANUAL_REVIEW: internal similarity score "
                f"{composite_score:.1f} is at or above the 60 auto-approve ceiling."
            )
            decision = "MANUAL_REVIEW"

    if notes:
        parsed["decision"] = decision
        parsed["safety_net_notes"] = notes
    return parsed


def _build_agent_input(
    vendor_payload: dict, match_result: dict | None, composite_score: float
) -> str:
    proposed_lines = "\n".join(
        f"  - {key}: {value}" for key, value in vendor_payload.items()
    )

    if match_result:
        match_block = (
            f"  - existing_vendor_id: {match_result.get('vendor_id')}\n"
            f"  - existing_vendor_name: {match_result.get('name')}\n"
            f"  - existing_vendor_website: {match_result.get('website')}\n"
            f"  - existing_vendor_country: {match_result.get('country_code')}\n"
            f"  - internal_similarity_score: {composite_score:.2f} / 100\n"
        )
    else:
        match_block = (
            "  - No internal vendor-master candidate matched closely enough to "
            "report; treat the similarity score as 0.\n"
        )

    return (
        "Proposed new vendor record (as submitted by the requester):\n"
        f"{proposed_lines}\n\n"
        "Best-matching existing vendor-master record (from the internal fuzzy + "
        "semantic matcher):\n"
        f"{match_block}\n"
        "Evaluate this proposed vendor using your routing rules. When calling "
        "your tools, use the proposed vendor's ORIGINAL legal name shown above "
        "(not a normalized or abbreviated form)."
    )


def _fallback_result(
    vendor_payload: dict, match_result: dict | None, composite_score: float
) -> dict[str, Any]:
    return {
        "decision": "MANUAL_REVIEW",
        "business_justification": "",
        "similarity_score": round(composite_score, 2),
        "sanctions_hit": None,
        "hierarchy_relationship_found": None,
        "related_parent_entity": None,
        "matched_existing_vendor_id": (
            match_result.get("vendor_id") if match_result else None
        ),
        "matched_existing_vendor_name": (
            match_result.get("name") if match_result else None
        ),
        "proposed_vendor_name": vendor_payload.get("name"),
    }


def run_validation(
    vendor_payload: dict[str, Any], match_result: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        composite_score = (
            float(match_result.get("composite_score", 0.0)) if match_result else 0.0
        )
    except (TypeError, ValueError):
        composite_score = 0.0

    result = _fallback_result(vendor_payload, match_result, composite_score)

    try:
        agent_input = _build_agent_input(vendor_payload, match_result, composite_score)
        raw_response = agent_executor.invoke({"input": agent_input})
    except Exception as exc:
        result["business_justification"] = (
            "Automated validation could not complete because the agent raised "
            f"an error ({exc}), so this request is routed to manual review as "
            "a precaution."
        )
        return result

    output_text = (
        raw_response.get("output", "") if isinstance(raw_response, dict) else ""
    )
    intermediate_steps = (
        raw_response.get("intermediate_steps", [])
        if isinstance(raw_response, dict)
        else []
    )
    evidence = _extract_tool_evidence(intermediate_steps)

    parsed = _safe_parse_json(output_text)
    if parsed is None or parsed.get("decision") not in VALID_DECISIONS:
        result["business_justification"] = (
            "Automated validation returned a response that could not be parsed "
            "as valid JSON with a recognized decision, so this request is "
            "routed to manual review as a precaution; the raw model output is "
            "preserved below for audit."
        )
        result["raw_agent_output"] = output_text
        result["sanctions_hit"] = evidence.get("sanctions_hit")
        result["hierarchy_relationship_found"] = evidence.get(
            "hierarchy_relationship_found"
        )
        result["related_parent_entity"] = evidence.get("related_parent_entity")
        return result

    parsed["similarity_score"] = round(composite_score, 2)
    parsed["matched_existing_vendor_id"] = (
        match_result.get("vendor_id") if match_result else None
    )
    parsed["matched_existing_vendor_name"] = (
        match_result.get("name") if match_result else None
    )
    parsed["proposed_vendor_name"] = vendor_payload.get("name")

    return _apply_safety_net(parsed, composite_score, evidence)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one vendor through the validation agent."
    )
    parser.add_argument("--name", default="EdgeVerve Systems Limited")
    parser.add_argument("--website", default="edgeverve.com")
    parser.add_argument("--country", default="India")
    parser.add_argument(
        "--master-csv",
        default=os.environ.get("VENDOR_MASTER_CSV", "novo_vendor_master.csv"),
    )
    args = parser.parse_args()

    vendor_payload = {
        "name": args.name,
        "website": args.website,
        "country": args.country,
    }

    match_result = None
    if os.path.exists(args.master_csv):
        from matcher import AlgorithmicMatcher

        algorithmic_matcher = AlgorithmicMatcher(master_db_path=args.master_csv)
        candidates = algorithmic_matcher.find_candidates(vendor_payload)
        match_result = candidates[0] if candidates else None
    else:
        print(
            f"[agent] '{args.master_csv}' not found -- skipping the internal matcher "
            f"and validating against GLEIF/compliance data alone (score treated as 0)."
        )

    outcome = run_validation(vendor_payload, match_result)
    print(json.dumps(outcome, indent=2))
