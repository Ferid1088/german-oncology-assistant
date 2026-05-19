ANALYZE_QUERY = """\
Du bist ein Assistent für deutsche S3-Onkologie-Leitlinien.
Analysiere die folgende Anfrage und antworte AUSSCHLIESSLICH mit JSON (kein Markdown, kein Text davor/danach):

{{
  "rewritten_query": "<präzise deutsche Suchanfrage für eine medizinische Vektordatenbank>",
  "guideline_id": "<'' | 'mamma' | 'krk' | 'lunge' | 'prosta'>",
  "grade": "<'' | 'A' | 'B' | '0'>",
  "chunk_type": "<'' | 'recommendation' | 'section'>",
  "intent": "<'factual' | 'recommendation' | 'comparison' | 'external'>"
}}

Regeln:
- rewritten_query: Formuliere als Schlüsselwörter für eine medizinische Fachsuche, keine vollständige Frage.
- guideline_id: Nur setzen wenn eine spezifische Leitlinie eindeutig aus der Anfrage hervorgeht.
- grade: Nur setzen wenn ein bestimmter Empfehlungsgrad explizit gefragt wird.
- intent 'external': Nur wenn die Frage eindeutig außerhalb des onkologischen Leitlinienbereichs liegt.

{history_block}Anfrage: {query}"""
