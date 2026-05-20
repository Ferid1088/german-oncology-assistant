import os
import json
from openai import OpenAI
from src.graph.state import RAGState
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool
from src.tools.compare_guidelines import compare_guidelines_tool
from src.tools.drug_class_lookup import drug_class_lookup_tool
from src.tools.calculate_bmi import calculate_bmi_tool
from src.tools.pubmed_search import pubmed_search_tool
from src.graph.permissions import is_source_allowed, is_tool_allowed
from src.prompts.agent import AGENT_SYSTEM

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
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _dispatch_tool(name: str, args: dict) -> str:
    raise RuntimeError("_dispatch_tool(state-aware) should be used instead")


def _dispatch_tool_with_state(state: RAGState, name: str, args: dict) -> str:
    if not is_tool_allowed(state, name):
        return json.dumps(
            {"error": f"Tool '{name}' ist für die Rolle '{state.get('user_role', 'user')}' nicht erlaubt."},
            ensure_ascii=False,
        )

    if name == "search_guidelines":
        results = search_guidelines_tool(**args)
        return json.dumps(results, ensure_ascii=False)
    if name == "lookup_empfehlung":
        result = lookup_empfehlung_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "compare_guidelines":
        result = compare_guidelines_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "drug_class_lookup":
        result = drug_class_lookup_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "calculate_bmi":
        result = calculate_bmi_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    if name == "pubmed_search":
        if not is_source_allowed(state, "pubmed"):
            return json.dumps(
                {"error": "Externe Quellen (PubMed) sind für diese Anfrage nicht erlaubt."},
                ensure_ascii=False,
            )
        result = pubmed_search_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


_FORCE_SEARCH = {"type": "function", "function": {"name": "search_guidelines"}}


def run_agent(state: RAGState, client: OpenAI | None = None) -> dict:
    """Tool-calling agent loop. First iteration always searches; second is optional."""
    c = client or _client()
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": state["rewritten_query"] or state["user_query"]},
    ]

    all_chunks: list[dict] = []
    tool_calls_log: list[dict] = []

    searched_queries: set[str] = set()

    for i in range(2):  # max iterations
        # Force a DB search on the first call; let the model decide on the second
        tool_choice = _FORCE_SEARCH if i == 0 else "auto"
        resp = c.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice=tool_choice,
        )
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

            result_str = _dispatch_tool_with_state(state, tc.function.name, args)
            tool_calls_log.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if tc.function.name == "search_guidelines":
                all_chunks.extend(json.loads(result_str))

        # No new searches were needed on this iteration — we're done
        if not made_new_search and i > 0:
            break

    return {
        "retrieved_chunks": all_chunks[:10],
        "tool_calls_log": tool_calls_log,
    }
