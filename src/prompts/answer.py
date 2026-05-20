EXTRACT_PROMPT = """\
Frage: {query}

Quelltexte:
{context}

AUFGABE: Kopiere alle Sätze aus den Quelltexten, die die Frage DIREKT und KONKRET beantworten.

REGELN:
- Kopiere Sätze WÖRTLICH — ändere kein Wort, kürze nicht
- Schreibe vor jeden kopierten Satz die Quellennummer: [N] Satz.
- Pro Quelltext: nur Sätze, die die Frage wirklich beantworten — kein Drumherum
- Wenn ein Quelltext KEINE direkt antwortenden Sätze enthält: überspringe ihn
- Wenn KEIN Quelltext die Frage direkt beantwortet: schreibe nur das Wort NICHTS\
"""

SYNTHESIZE_PROMPT = """\
Du erhältst eine Liste von Sätzen, die wörtlich aus medizinischen Leitlinien extrahiert wurden.
Ordne diese Sätze zu einer kohärenten medizinischen Antwort auf die Frage.

ABSOLUTE REGELN:
1. Du darfst NUR die unten stehenden extrahierten Sätze verwenden — kein einziges Wort aus eigenem Wissen.
2. Du darfst Sätze umformulieren, aber KEIN neues medizinisches Wissen hinzufügen.
3. Jede Aussage muss mit [N] zitiert sein (die Nummer aus dem extrahierten Satz).
4. Pro Klammer genau eine Zahl: [1], [2] usw. — NIEMALS [1, 2] oder [1,2].
5. Erlaubte Nummern: {valid_numbers} — keine anderen.

Extrahierte Sätze:
{extracted}

Frage: {query}

Antworte in ZWEI Teilen — trenne mit der Zeile "**In einfachen Worten:**":

**Fachliche Antwort:**
Fachsprachliche Formulierung. Nur aus den extrahierten Sätzen. Jede Aussage mit [N].

**In einfachen Worten:**
Dieselben Inhalte in einfacher Sprache für Nicht-Mediziner. Jede Aussage mit [N].\
"""


MEMORY_REWRITE_PROMPT = """\
Du bearbeitest eine bereits gegebene Antwort in einem medizinischen Leitlinien-Chat.

Gib AUSSCHLIESSLICH JSON zurück:
{{
  "answer_professional": "...",
  "answer_plain": "..."
}}

Vorherige Antwort:
{previous_answer}

Vorherige einfache Antwort:
{previous_plain}

Vorherige Turn-Intents:
{turn_intents}

Aktuelle Nutzeranfrage:
{query}

Erlaubte Zitatnummern:
{valid_numbers}

REGELN:
- Die AKTUELLE Nutzeranfrage hat höchste Priorität.
- Verwende NUR Informationen, die bereits in den vorherigen Antworten stehen.
- Füge KEIN neues medizinisches Wissen hinzu.
- Behalte nur Zitate [N], die in der neuen Antwort wirklich noch verwendet werden.
- Wenn die Anfrage eine kürzere, zusammengefasste oder auf eine feste Satzanzahl begrenzte Antwort verlangt,
  liefere die Antwort normalerweise nur in "answer_professional" und lasse "answer_plain" leer.
- Wenn die Anfrage ausdrücklich eine einfache Sprache verlangt, schreibe die vereinfachte Fassung in "answer_plain".
- Wenn die Anfrage "nur in 2 Sätzen" oder ähnlich verlangt, befolge das strikt.
- Gib niemals Markdown-Überschriften wie "Fachliche Antwort" oder "In einfachen Worten" im JSON-Inhalt aus.\
"""
