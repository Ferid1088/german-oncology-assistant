TURN_ROUTER_PROMPT = """\
Du klassifizierst die aktuelle Nutzeranfrage in einem medizinischen Leitlinien-Chat.

Gib AUSSCHLIESSLICH JSON zurück:
{{
  "turn_intents": ["clarify", "simplify", "expand", "refine", "new_query"],
  "followup_routing": "memory" | "retrieve"
}}

Regeln:
- Mehrfachklassifikation ist erlaubt und erwünscht, wenn mehrere Absichten gleichzeitig vorliegen.
- Erlaube sinnvolle Kombinationen wie:
  - ["clarify", "simplify"]
  - ["clarify", "refine"]
  - ["simplify", "expand"]
  - ["expand", "refine"]
- Verwende "new_query", wenn eine neue Frage oder ein Themenwechsel vorliegt.
- followup_routing = "memory", wenn die Anfrage voraussichtlich mit dem bisherigen Gespräch,
  der letzten Antwort oder den zuletzt verwendeten Leitlinienzitaten/chunks beantwortet werden kann.
- followup_routing = "retrieve", wenn neue Leitlinieninhalte gesucht werden sollten.
- Wenn "new_query" enthalten ist, wähle normalerweise "retrieve".
- Wenn die Nutzeranfrage nur um Erklärung, Vereinfachung, Umformulierung oder kurze Ausführung bittet,
  ohne das Thema zu ändern, wähle normalerweise "memory".

Letzte Konversation:
{history_block}

Aktuelle Anfrage:
{query}
"""