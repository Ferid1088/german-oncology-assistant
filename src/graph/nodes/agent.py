"""Agent node: GPT-4o tool-calling loop that drives guideline retrieval.

Runs a maximum of 2 iterations:
- **Iteration 1** — ``search_guidelines`` is *forced* via ``tool_choice`` so that
  every query always starts with at least one vector search, regardless of what
  the LLM might otherwise decide.
- **Iteration 2** — ``tool_choice`` is set to ``"auto"``, allowing GPT-4o to call
  any tool (lookup, compare, drug search, BMI, PubMed) or stop if the first
  iteration already gathered sufficient context.

A deduplication guard (``searched_queries`` set) prevents GPT-4o from issuing
the same ``search_guidelines`` query twice, which would waste tokens and add no
new information.

The node is bypassed entirely when ``followup_routing == "memory"``, in which case
the prior turn's chunks are returned directly without any database access.
"""

import os
import json
import time
from openai import OpenAI
from src.graph.state import RAGState
from src.retrieval.postprocess import top_unique_result_dicts
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool
from src.tools.compare_guidelines import compare_guidelines_tool
from src.tools.drug_class_lookup import drug_class_lookup_tool
from src.tools.calculate_bmi import calculate_bmi_tool
from src.tools.pubmed_search import pubmed_search_tool
from src.graph.permissions import is_source_allowed, is_tool_allowed
from src.prompts.agent import AGENT_SYSTEM
from src.telemetry import append_rag_step, merge_token_usage, summarize_tool_result, usage_from_response

GEN_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_guidelines",
            "description": "Suche in deutschen Onkologie-Leitlinien nach relevantem Inhalt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "guideline_id": {"type": "string", "enum": ["mamma", "krk", "lunge", "prosta", ""]},
                    "grade": {"type": "string", "enum": ["A", "B", "0", ""]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_empfehlung",
            "description": "Rufe eine spezifische Empfehlung X.Y direkt ab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "guideline_id": {"type": "string"},
                    "recommendation_id": {"type": "string"},
                },
                "required": ["guideline_id", "recommendation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_guidelines",
            "description": "Vergleiche zwei Leitlinien zu einem Thema anhand der Datenbankfunde.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "guideline_a": {"type": "string", "enum": ["mamma", "krk", "lunge", "prosta"]},
                    "guideline_b": {"type": "string", "enum": ["mamma", "krk", "lunge", "prosta"]},
                },
                "required": ["topic", "guideline_a", "guideline_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drug_class_lookup",
            "description": "Suche Leitlinien-Erwähnungen zu einem Medikament oder Wirkstoff über mehrere Leitlinien.",
            "parameters": {
                "type": "object",
                "properties": {
                    "substance_name": {"type": "string"},
                },
                "required": ["substance_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Berechne den BMI aus Gewicht und Größe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number"},
                    "height_cm": {"type": "number"},
                },
                "required": ["weight_kg", "height_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Suche externe Literatur in PubMed, wenn Leitlinienmaterial nicht ausreicht.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]


def _client() -> OpenAI:
    """Return an OpenRouter-backed OpenAI client."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _dispatch_tool(name: str, args: dict) -> str:
    # Kept as a guard — callers must use _dispatch_tool_with_state so that
    # RBAC checks are always applied before tool execution.
    raise RuntimeError("_dispatch_tool(state-aware) should be used instead")


def _dispatch_tool_with_state(state: RAGState, name: str, args: dict) -> tuple[str, object]:
    """Execute a tool call after verifying RBAC permissions.

    Checks ``is_tool_allowed()`` first; if the role does not permit the tool,
    returns an error JSON string without calling the underlying function.
    For ``pubmed_search``, also verifies that ``"pubmed"`` is in ``allowed_sources``.

    Args:
        state: Current RAGState used for RBAC checks.
        name: Tool function name as declared in ``TOOLS_SPEC``.
        args: Keyword arguments parsed from the LLM's function call JSON.

    Returns:
        A tuple of ``(json_string, parsed_result)`` where ``json_string`` is
        appended to the message history and ``parsed_result`` is used for
        chunk collection and logging.
    """
    if not is_tool_allowed(state, name):
        result = {"error": f"Tool '{name}' ist für die Rolle '{state.get('user_role', 'user')}' nicht erlaubt."}
        return json.dumps(result, ensure_ascii=False), result

    if name == "search_guidelines":
        results = search_guidelines_tool(**args)
        return json.dumps(results, ensure_ascii=False), results
    if name == "lookup_empfehlung":
        result = lookup_empfehlung_tool(**args)
        return json.dumps(result, ensure_ascii=False), result
    if name == "compare_guidelines":
        result = compare_guidelines_tool(**args)
        return json.dumps(result, ensure_ascii=False), result
    if name == "drug_class_lookup":
        result = drug_class_lookup_tool(**args)
        return json.dumps(result, ensure_ascii=False), result
    if name == "calculate_bmi":
        result = calculate_bmi_tool(**args)
        return json.dumps(result, ensure_ascii=False), result
    if name == "pubmed_search":
        if not is_source_allowed(state, "pubmed"):
            result = {"error": "Externe Quellen (PubMed) sind für diese Anfrage nicht erlaubt."}
            return json.dumps(result, ensure_ascii=False), result
        result = pubmed_search_tool(**args)
        return json.dumps(result, ensure_ascii=False), result
    result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False), result


# Forces search_guidelines on iteration 1 regardless of GPT-4o's preference.
# Without this the model might skip retrieval for seemingly simple questions.
_FORCE_SEARCH = {"type": "function", "function": {"name": "search_guidelines"}}


def run_agent(state: RAGState, client: OpenAI | None = None) -> dict:
    """Tool-calling agent loop. Retrieval can be skipped for memory-routed follow-ups."""
    if state.get("followup_routing") == "memory":
        return {
            "retrieved_chunks": state.get("prior_retrieved_chunks", []),
            "tool_calls_log": state.get("tool_calls_log", []),
            "rag_trace": append_rag_step(
                state.get("rag_trace", []),
                name="agent",
                status="skipped",
                summary="Agent retrieval was skipped because the turn reused prior conversation memory.",
            ),
        }

    c = client or _client()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": state["rewritten_query"] or state["user_query"]},
    ]

    all_chunks: list[dict] = []
    tool_calls_log: list[dict] = []
    token_usage = state.get("token_usage", {})

    searched_queries: set[str] = set()

    for i in range(2):  # max iterations
        tool_choice = _FORCE_SEARCH if i == 0 else "auto"
        started = time.perf_counter()
        resp = c.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice=tool_choice,
            max_tokens=1500,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        token_usage = merge_token_usage(token_usage, usage_from_response(resp, model=GEN_MODEL, step=f"agent_iteration_{i + 1}", duration_ms=duration_ms))
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        made_new_search = False
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)

            # Skip duplicate search_guidelines calls with the same query
            if tc.function.name == "search_guidelines":
                q_key = args.get("query", "").strip().lower()
                if q_key in searched_queries:
                    continue
                searched_queries.add(q_key)
                made_new_search = True

            result_str, parsed_result = _dispatch_tool_with_state(state, tc.function.name, args)
            summary, preview, count, status = summarize_tool_result(tc.function.name, parsed_result)
            tool_calls_log.append(
                {
                    "tool": tc.function.name,
                    "args": args,
                    "summary": summary,
                    "preview": preview,
                    "result_count": count,
                    "status": status,
                }
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if tc.function.name == "search_guidelines":
                if isinstance(parsed_result, list):
                    all_chunks.extend(parsed_result)

        # No new searches were needed on this iteration — we're done
        if not made_new_search and i > 0:
            break

    top_chunks = top_unique_result_dicts(all_chunks, top_k=10)
    return {
        "retrieved_chunks": top_chunks,
        "tool_calls_log": tool_calls_log,
        "token_usage": token_usage,
        "rag_trace": append_rag_step(
            state.get("rag_trace", []),
            name="agent",
            status="ok" if top_chunks else "empty",
            summary=f"Agent completed retrieval with {len(top_chunks)} unique chunk(s).",
            details={"tool_calls": len(tool_calls_log), "retrieved_chunks": len(top_chunks)},
        ),
    }
