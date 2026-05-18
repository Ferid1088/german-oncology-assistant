import os
import json
from openai import OpenAI
from src.graph.state import RAGState
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool

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
]


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _dispatch_tool(name: str, args: dict) -> str:
    if name == "search_guidelines":
        results = search_guidelines_tool(**args)
        return json.dumps(results, ensure_ascii=False)
    if name == "lookup_empfehlung":
        result = lookup_empfehlung_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(state: RAGState, client: OpenAI | None = None) -> dict:
    """Tool-calling agent loop. Runs until the model stops calling tools."""
    c = client or _client()
    messages = [
        {"role": "system", "content": (
            "Du bist ein medizinischer Leitlinien-Assistent für deutsche S3-Onkologie-Leitlinien. "
            "Nutze die verfügbaren Tools um relevante Leitlinienabschnitte zu finden, bevor du antwortest."
        )},
        {"role": "user", "content": state["rewritten_query"] or state["user_query"]},
    ]

    all_chunks: list[dict] = []
    tool_calls_log: list[dict] = []

    for _ in range(5):  # max iterations
        resp = c.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result_str = _dispatch_tool(tc.function.name, args)
            tool_calls_log.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if tc.function.name == "search_guidelines":
                all_chunks.extend(json.loads(result_str))

    return {
        "retrieved_chunks": all_chunks[:10],
        "tool_calls_log": tool_calls_log,
    }
