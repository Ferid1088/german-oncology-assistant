# German Onkologie Assistent

Ein Retrieval-Augmented-Generation-System (RAG) zur Abfrage deutscher S3-Leitlinien in der Onkologie. Klinische Fragen werden auf Deutsch gestellt; das System sucht belegbasierte Antworten aus vier Krebsleitlinien und generiert professionelle sowie laienverständliche Antworten mit vollständigen Quellenangaben.

---

## Übersicht

Das System basiert auf einer **LangGraph-Zustandsmaschine** mit 12 Knoten. Jede Anfrage durchläuft Eingabe-Sicherheitsprüfungen, Query-Umformulierung, intelligentes Routing, einen GPT-4o-Tool-Calling-Agenten, Konfidenzprüfung, Antwortgenerierung und Ausgabe-Sicherheitsprüfungen – bevor eine zitierte, belegte Antwort zurückgegeben wird.

**Unterstützte Leitlinien:**

| ID | Leitlinie | Version |
|---|---|---|
| `mamma` | Mammakarzinom | S3 v4.4 |
| `krk` | Kolorektales Karzinom | S3 v3.0 |
| `lunge` | Lungenkarzinom | S3 v4.0 |
| `prosta` | Prostatakarzinom | S3 v8.0 |

---

## Architektur

### Haupt-Pipeline

```
                              Benutzeranfrage
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       GUARDRAIL INPUT        │
                   │  kein LLM · regelbasiert     │
                   │  ① Prompt-Injection-Erkennung│──── blockiert ──► [BLOCKIERT] ──► ENDE
                   │  ② Off-Topic-Keyword-Block   │
                   │  ③ PII-Schwärzung (kein Bl.) │
                   └──────────────┬───────────────┘
                                  │ weiter
                                  ▼
                   ┌──────────────────────────────┐
                   │            REWRITE           │
                   │  Gemini 2.5 Flash · 400 Tok  │──── Rückfrage ──► [KLÄRUNG]       ──► ENDE
                   │  → rewritten_query           │
                   │  → Filter {Leitlinie, Grad}  │──── Duplikat ───► [ANTWORT WDHL.] ──► ENDE
                   │  → Intent-Klassifikation     │
                   │  → Query-Dekomposition       │
                   └──────────────┬───────────────┘
                                  │ ok
                                  ▼
                   ┌──────────────────────────────┐
                   │          TURN ROUTER         │
                   │  ① Heuristik (kein LLM)      │
                   │    vereinfach→simplify        │
                   │    kürzer→shorten  usw.       │
                   │  ② Gemini (nur bei Bedarf)   │
                   │  → followup_routing:         │
                   │    "retrieve" | "memory"     │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │             AGENT            │──── memory ──► vorherige Chunks wiederverwenden
                   │  GPT-4o · max. 2 Iterationen │
                   │  Iter. 1: ERZWUNGENE Suche   │
                   │  Iter. 2: auto (beliebig)    │
                   │  RBAC je Tool-Aufruf geprüft │
                   └──────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
 │search_guidelines│   │lookup_empfehlung │   │ compare_guidelines  │
 │ Hybridsuche     │   │ exakter Filter   │   │ Suche × 4 Leitl.    │
 │ (s. unten)      │   │ recommendation_id│   │ Ergebnisse gruppiert│
 └────────┬────────┘   └────────┬─────────┘   └──────────┬──────────┘
          │                     │                         │
          ▼                     ▼                         ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
 │drug_class_lookup│   │  calculate_bmi   │   │   pubmed_search     │
 │ Wirkstoffsuche  │   │ BMI = kg/m²      │   │ esearch → esummary  │
 │ über 4 Leitl.   │   │ + WHO-Kategorie  │   │ 5 strukturierte Erg.│
 └────────┬────────┘   └────────┬─────────┘   └──────────┬──────────┘
          └───────────────────────┴───────────────────────┘
                                  │  retrieved_chunks
                                  ▼
                   ┌──────────────────────────────┐
                   │          CONFIDENCE          │
                   │  Score = Ø(top-3 Reranker)   │──── Score<0,5 ──────────────────────────┐
                   │  Schwelle: 0,5               │          oder                           │
                   │  Min. Chunks: 2              │──── Chunks<2  ──────────────────────────┤
                   └──────────────┬───────────────┘                                        │
                                  │ ok                                          ┌───────────▼──────────┐
                                  │                                             │        ESCALATE      │
                                  │                                             │  4 Query-Varianten:  │
                                  │                                             │  · rewritten_query   │
                                  │                                             │  · Dekompositions-   │
                                  │                                             │    teile             │
                                  │                                             │  · "Leitlinienempf.."│
                                  │◄────────────────────────────────────────────  → Top-10 gemergt    │
                                  ▼                                             └──────────────────────┘
                   ┌──────────────────────────────┐
                   │             ANSWER           │
                   │  Pfad A — neue Abfrage       │
                   │  ┌──────────────────────┐   │
                   │  │ ① Gemini EXTRAKT     │   │
                   │  │   wörtl. [N]-Tags    │   │
                   │  │   "NICHTS" → ② überspr│  │
                   │  └──────────┬───────────┘   │
                   │             ▼               │
                   │  ┌──────────────────────┐   │
                   │  │ ② GPT-4o SYNTHESE    │   │
                   │  │   {professionell,    │   │
                   │  │    laienverständl.}  │   │
                   │  │   + Zitate + HINWEIS │   │
                   │  └──────────────────────┘   │
                   │  Pfad B — Gedächtnis         │
                   │  Heuristik (kürzen/vereinf.) │
                   │  oder GPT-4o-Umschreibung    │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       GUARDRAIL OUTPUT       │
                   │  kein LLM · regelbasiert     │──── blockiert ──► ENDE  (Sicherheitsfelder gesetzt)
                   │  ① Patientenspezifisch-Erk.  │
                   │  ② Dosierungs-Erkennung      │
                   │  ③ Belegungs-Prüfung         │
                   │    (hebt Block auf, wenn bel.)│
                   └──────────────┬───────────────┘
                                  │ weiter
                                  ▼
                   ┌──────────────────────────────┐
                   │        EXTERNAL SEARCH       │
                   │  Google CSE → DuckDuckGo     │
                   │  → [] bei Fehler             │
                   │  übersprungen wenn: blockiert│
                   │   / Klärung / kein Web-Perm. │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                                 ENDE
                         SSE → Streamlit-UI
                   data: { answer, citations, tool_calls,
                           rag_trace, token_usage, trace_id }
                   data: [DONE]
```

### Hybrid-Suche *(aufgerufen durch `search_guidelines`)*

```
          Suchbegriff
               │
               ▼
   ┌───────────────────────┐
   │  text-embedding-3-    │   3072 Dim. · Batch 64
   │  large  (OpenAI)      │
   └─────────┬─────────────┘
             │  Query-Vektor
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌──────────┐   ┌──────────────┐   ┌──────────────────────┐
│ Milvus   │   │   Milvus     │   │   BM25 Sparse        │
│ ANN #1   │   │   ANN #2     │   │   rank_bm25.         │
│ alle Typen│  │ nur Empf.    │   │   OkapiBM25          │
│ top-20   │   │ top-10       │   │   bm25_index.pkl     │
│ COSINE   │   │ COSINE/HNSW  │   │   top-20 Treffer     │
│ HNSW     │   │              │   │                      │
└────┬─────┘   └──────┬───────┘   └──────────┬───────────┘
     │                │                       │
     └────────┬───────┘                       │
              ▼                               │
   ┌─────────────────────┐                    │
   │   RRF  Runde 1      │                    │
   │  fuse(ANN#1, ANN#2) │                    │
   │  k = 60             │                    │
   └──────────┬──────────┘                    │
              └──────────────┬────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   RRF  Runde 2      │
                  │ fuse(Runde1, BM25)  │
                  │ k = 60              │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   CrossEncoder      │
                  │ BAAI/bge-reranker   │
                  │ -v2-m3 (lokal)      │
                  │ bis 30 Kandidaten   │
                  │ Fallback: RRF-Rang  │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Parent Expansion   │
                  │ Parent-Text voranst.│
                  │ Seitenb. zusammenf. │
                  │ lautlos bei Fehler  │
                  └──────────┬──────────┘
                             ▼
                      top-5 Chunks
                   (RetrievedChunk[])
```

### LLM-Aufrufe pro Anfrage *(Normalpfad)*

```
  Knoten          Modell                 Zweck                    Token
  ──────────────────────────────────────────────────────────────────────
  rewrite         Gemini 2.5 Flash       Query erweitern/klassif.   400
  turn_router     Gemini 2.5 Flash       Intent-Klassifikation      200  ← nur bei Heuristik-Fehler
  agent Iter. 1   GPT-4o                 Erzwungener Suchaufruf      —
  agent Iter. 2   GPT-4o                 Optionale weitere Tools     —
  answer extract  Gemini 2.5 Flash       Wörtliche Extraktion      1500
  answer synth    GPT-4o                 Professionell + Laien-JSON 1200
  ──────────────────────────────────────────────────────────────────────
  Alle über OpenRouter  (ein API-Schlüssel · eine Basis-URL)
```

## Indexing Pipeline  *(offline · `python scripts/run_indexer.py`)*

```
   PDF  (docs/knowledge_base/)
        │
        ▼
   ┌─────────┐     extract pages · detect printed page numbers
   │  PARSE  │     skip TOC · strip copyright header
   └────┬────┘     repair hyphenation · merge continuation lines
        │
        ▼
   ┌──────────┐    single-pass state machine → StructuralUnit:
   │ DETECT   │    heading | empfehlung | bibliography | prose
   └────┬─────┘
        │
        ▼
   ┌───────┐       heading    → parent Chunk  (is_leaf=False)
   │ CHUNK │       empfehlung → leaf   Chunk  (chunk_type=recommendation)
   └───┬───┘       prose      → sliding-window leaves
        │               TARGET_TOKENS=550 · OVERLAP_TOKENS=70
        ▼
   ┌──────────┐    attach: source_filename · is_current
   │ METADATA │    assign: chunk_index_in_parent
   └────┬─────┘
        │
        ▼
   ┌────────┐      3 LLM calls per chunk  (Gemini 2.5 Flash)
   │ ENRICH │  ①  contextual header  (prepended before embed)
   │optional│  ②  hypothetical questions  (HyDE)
   └────┬───┘  ③  semantic metadata  {diseases, drugs, …}
        │
        ▼
   ┌───────┐       text-embedding-3-large · 3072 dim
   │ EMBED │       input: header + questions + chunk_text
   └───┬───┘       batch size: 64
        │
        ▼
   ┌────────┐      Milvus Lite  (./milvus.db)
   │ UPSERT │      HNSW: M=16 · efConstruction=200 · COSINE
   └────┬───┘      dynamic fields for enricher metadata
        │
        ▼
   ┌─────────────┐  query all is_leaf chunks (paginated 1000/page)
   │ BM25 REBUILD│  build OkapiBM25 → bm25_index.pkl
   └─────────────┘  reload in-process singleton
```

---

## Funktionen

### Kernfunktionen
- **Hybrid-RAG-Suche:** Dichte Vektoren (Milvus HNSW) + BM25 Sparse, fusioniert per Reciprocal Rank Fusion
- **CrossEncoder-Reranking:** `BAAI/bge-reranker-v2-m3` bewertet die Top-12 Kandidaten neu
- **6 Domänen-Tools:** Leitliniensuche, Empfehlungsabruf, Leitlinienvergleich, Wirkstoffsuche, BMI-Berechnung, PubMed-Suche
- **Zweistufige Antwortgenerierung:** Gemini 2.5 Flash extrahiert wörtliche Sätze → GPT-4o synthetisiert professionelle + laienverständliche Antwort
- **Vollständige Zitatkette:** `[1]`, `[2]` als Inline-Referenzen mit Seite, Abschnitt und Empfehlungsgrad
- **Gesprächsgedächtnis:** Wiederholte oder Folge-Anfragen verwenden frühere Antworten ohne erneuten Datenbankzugriff

### Sicherheit
- **Eingabe-Guardrail:** Blockiert Prompt-Injections und Off-Topic-Anfragen; schwärzt PII (Daten, Namen, Telefonnummern, Postleitzahlen) vor jedem LLM-Zugriff
- **Ausgabe-Guardrail:** Blockiert Antworten ohne Belege; schränkt Dosierungsangaben ein, wenn nicht direkt durch abgerufene Chunks belegt; fügt patientenspezifische Hinweise hinzu
- **Konfidenz-Eskalation:** Niedriger Reranker-Score löst eine 4-fache Multi-Strategie-Fallback-Suche aus

### Infrastruktur
- **Rate-Limiting:** Gleitendes Fenster, pro Benutzer und Route (20 Anf./60s für Chat)
- **API-Schlüsselverwaltung:** Unterstützung mehrerer Schlüssel über Umgebungsvariablen
- **Strukturiertes Logging:** JSON-Logs mit Trace-IDs und Anfragedauer
- **Token-Tracking:** Verbrauch und Kosten (USD) je Aufruf, aggregiert über die gesamte Pipeline
- **Gesprächspersistenz:** SQLite-basierter Gesprächsverlauf mit Session-Isolierung

### Benutzeroberfläche
- **Streamlit-Chat-Oberfläche** mit Echtzeit-SSE-Streaming
- **Quellenkarten:** Abgerufene Chunks mit Leitlinie, Abschnitt, Seitenbereich, Grad und Evidenzlevel
- **RAG-Prozessvisualisierung:** Schritt-für-Schritt-Trace mit Status und Dauer je Pipeline-Knoten
- **Analytics-Dashboard:** Token-Verbrauch, Kosten, Tool-Häufigkeit, Leitlinienverteilung, Zeitreihen
- **Export:** Gespräche als JSON, CSV oder PDF herunterladen

---

## Voraussetzungen

- Python 3.12+
- Ein [OpenRouter](https://openrouter.ai)-API-Schlüssel (routet zu GPT-4o und Gemini 2.5 Flash)
- PDF-Dateien der S3-Leitlinien (nicht im Repository enthalten)

---

## Installation

```bash
git clone https://github.com/Ferid1088/german-oncology-assistant.git
cd german-oncology-assistant

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e .
pip install -e ".[dev]"         # Entwicklungsabhängigkeiten (Tests)
```

---

## Konfiguration

```bash
cp .env.example .env
```

`.env`-Referenz:

```env
# Erforderlich — Schlüssel unter https://openrouter.ai
OPENROUTER_API_KEY=your_key_here

# LLM-Modelle
GENERATION_MODEL=openai/gpt-4o
CHEAP_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=openai/text-embedding-3-large

# Milvus — leer lassen für Milvus Lite (./milvus.db, kein Server erforderlich)
MILVUS_URI=
MILVUS_COLLECTION=oncology_guidelines

# API-Authentifizierung — in Produktion ändern
API_KEY=dev-secret-key
# Mehrere Schlüssel kommagetrennt:
# API_KEYS=key1,key2,key3

# Gesprächsdatenbank (SQLite)
CONVERSATION_DB_PATH=data/app_state.db

# Optional: Google Custom Search für externe Webergebnisse
# GOOGLE_SEARCH_API_KEY=
# GOOGLE_SEARCH_ENGINE_ID=

# Protokollierungsstufe: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO

# Optional: PostgreSQL für LangGraph-Checkpointing
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/oncology_rag
```

---

## Leitlinien indizieren

Vor dem Start müssen die PDF-Leitlinien in die Milvus-Vektordatenbank indiziert werden. PDFs im Verzeichnis `data/` ablegen und ausführen:

```bash
python scripts/run_indexer.py                        # Alle 4 Leitlinien
python scripts/run_indexer.py --pdf mammakarzinom_v4.4.pdf  # Einzelne Leitlinie
python scripts/run_indexer.py --dry-run              # Nur parsen, kein Datenbankschreiben
python scripts/run_indexer.py --no-enrich            # Schneller, ohne LLM-Anreicherung
```

Mit aktivierter Anreicherung (Standard) ruft Gemini 2.5 Flash je Chunk folgendes auf:
- Kontextuellen Header (klinischer Kontext und Inhalt)
- 2–3 hypothetische Fragen, die ein Kliniker zu diesem Chunk stellen könnte
- Semantische Metadaten (Erkrankungen, Medikamente, Verfahren, Patientengruppen)

**Erwartete Dateinamen** (konfiguriert in `src/indexer/pipeline.py`):

| Datei | Leitlinien-ID |
|---|---|
| `mammakarzinom_v4.4.pdf` | `mamma` |
| `kolorektales_v3.0.pdf` | `krk` |
| `lungenkarzinom_v4.0.pdf` | `lunge` |
| `prostatakarzinom_v8.0.pdf` | `prosta` |

---

## Anwendung starten

```bash
python scripts/run_app.py
```

| Dienst | URL |
|---|---|
| Streamlit-UI | http://localhost:8501 |
| FastAPI-Backend | http://localhost:8000 |
| API-Dokumentation (Swagger) | http://localhost:8000/docs |
| Health-Check | http://localhost:8000/health |

Dienste einzeln starten:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run src/ui/app.py --server.port 8501
```

---

## API-Referenz

Alle Endpunkte erfordern den Header `X-API-Key` (Wert aus der Umgebungsvariable `API_KEY`).

### Chat

```http
POST /chat
Content-Type: application/json
X-API-Key: dev-secret-key

{
  "query": "Welche Erstlinientherapie empfiehlt die S3-Leitlinie beim HER2+ Mammakarzinom?",
  "session_id": "my-session-123",
  "guideline_id": "mamma",   // optional: mamma | krk | lunge | prosta
  "grade": "A"               // optional: A | B | 0
}
```

Antwort wird als Server-Sent Events (SSE) gestreamt. Das finale Event enthält `answer_professional`, `answer_plain`, `citations`, `rag_trace` und `token_usage`.

### Gespräche

```http
GET    /conversations                             # Alle Sessions auflisten
GET    /conversations/{session_id}                # Session laden
DELETE /conversations/{session_id}                # Session löschen
POST   /conversations/{session_id}/export?format=json   # Export (json | csv | pdf)
```

### Analytics

```http
GET /analytics/overview    # Token-Verbrauch, Kosten, Tool-Häufigkeit, Leitlinienverteilung
```

---

## Knotenübersicht (12 Knoten)

**Gruppe 1 — 8 Knoten mit eigenen Dateien in `src/graph/nodes/`:**

| Knoten | Datei | Modell | Funktion |
|---|---|---|---|
| `guardrail_input` | `guardrail_input.py` | — | Regex: blockiert Injections, Off-Topic; schwärzt PII |
| `rewrite` | `rewriter.py` | Gemini 2.5 Flash | Normalisiert Query, extrahiert Filter, erkennt Mehrdeutigkeit |
| `turn_router` | `turn_router.py` | Gemini 2.5 Flash | Klassifiziert Turn-Intent; routet zu Gedächtnis oder Abruf |
| `agent` | `agent.py` | GPT-4o | 2-Iterations-Tool-Calling-Loop; Iter. 1 ruft immer `search_guidelines` auf |
| `confidence` | `confidence.py` | — | Ø Reranker-Score Top-3; eskaliert bei Score < 0,5 |
| `answer` | `answer.py` | Gemini + GPT-4o | Stufe 1: wörtliche Extraktion (Gemini). Stufe 2: Synthese (GPT-4o) |
| `guardrail_output` | `guardrail_output.py` | — | Blockiert unbelegte Antworten; schränkt Dosierung ein |
| `external_search` | `external_search.py` | — | Google/DuckDuckGo ergänzende Snippets |

**Gruppe 2 — 4 Knoten als private Funktionen in `src/graph/graph.py`:**

| Knoten | Funktion | Funktion |
|---|---|---|
| `blocked` | `_blocked_response()` | Gibt `input_block_reason` als finale Antwort zurück |
| `clarification` | `_clarification_response()` | Gibt eine deutsche Rückfrage zurück |
| `repeat_answer` | `_repeat_previous_answer_response()` | Gibt vorherige Antwort unverändert zurück |
| `escalate` | `_multi_query_escalation()` | Generiert 4 Query-Varianten und führt Fallback-Suche durch |

---

## Wichtige Designentscheidungen

**Warum LangGraph?**
Jeder Pipeline-Schritt ist unabhängig testbar, nachverfolgbar und austauschbar. Jeder Knoten liest aus und schreibt in ein einzelnes `RAGState`-TypedDict.

**Warum Hybrid-Suche?**
Dichte Vektoren erfassen semantische Ähnlichkeit; BM25 trifft exakte Keyword-Treffer (Wirkstoffnamen, Empfehlungs-IDs, Abschnittsnummern). RRF-Fusion kombiniert beides ohne kalibrierte Scores.

**Warum zweifache Dense-Suche?**
Eine einzelne Suche über alle Chunk-Typen lässt lange Prosa-Abschnitte kurze Empfehlungs-Chunks verdrängen. Die zweite Suche nur auf `chunk_type=recommendation` stellt sicher, dass klinische Empfehlungen immer im Kandidatenpool erscheinen.

**Warum zweistufige Antwortgenerierung?**
Stufe 1 (Gemini-Extraktion) zwingt das Modell, Sätze wörtlich aus dem abgerufenen Text zu kopieren – kein Paraphrasieren, kein Trainingswissen. Stufe 2 (GPT-4o-Synthese) formuliert nur um, was bereits extrahiert wurde. Halluzinationen werden strukturell erschwert.

**Warum kein LLM im Eingabe-Guardrail?**
Ein LLM zur Erkennung von Prompt-Injections schafft eine zirkuläre Angreifbarkeit. Regex und Keyword-Matching sind deterministisch und können nicht durch Jailbreaks umgangen werden.

---

## Projektstruktur

```
.
├── src/
│   ├── api/                    FastAPI-Backend
│   │   ├── main.py             Einstiegspunkt, Middleware
│   │   ├── routes/             chat.py, conversations.py, analytics.py
│   │   ├── auth.py             API-Schlüssel-Verifizierung
│   │   ├── rate_limit.py       Gleitendes-Fenster-Rate-Limiter
│   │   ├── observability.py    JSON-Logging, Trace-IDs
│   │   ├── conversation_store.py  SQLite-Persistenz
│   │   ├── export_utils.py     JSON / CSV / PDF Export
│   │   └── analytics_service.py
│   │
│   ├── graph/                  LangGraph-Zustandsmaschine
│   │   ├── graph.py            build_graph() — 12-Knoten-StateGraph
│   │   ├── state.py            RAGState TypedDict (~35 Felder)
│   │   ├── permissions.py      RBAC: is_tool_allowed(), is_source_allowed()
│   │   └── nodes/              8 Knoten-Module (s. oben)
│   │
│   ├── retrieval/              Suche und Ranking
│   │   ├── search.py           hybrid_search() — Dense + BM25 + RRF
│   │   ├── bm25.py             BM25-Index Aufbau und Laden
│   │   ├── reranker.py         CrossEncoder-Reranking
│   │   ├── expander.py         Parent-Chunk-Erweiterung
│   │   └── postprocess.py      Deduplizierung
│   │
│   ├── tools/                  Agent-Tools (6 + Web)
│   │   ├── search_guidelines.py
│   │   ├── lookup_empfehlung.py
│   │   ├── compare_guidelines.py
│   │   ├── drug_class_lookup.py
│   │   ├── calculate_bmi.py
│   │   ├── pubmed_search.py
│   │   └── web_search.py
│   │
│   ├── indexer/                PDF-Ingestion-Pipeline
│   │   ├── pipeline.py         index_pdf() — Haupt-Orchestrierung
│   │   ├── chunker.py          Hierarchisches Chunking (550 Tok, 70 Überlappung)
│   │   ├── embedder.py         embed_texts() — Batch 64
│   │   ├── store.py            MilvusStore — HNSW-Collection-Management
│   │   ├── enricher.py         LLM-Anreicherung (Header, HyDE-Fragen, Metadaten)
│   │   ├── metadata.py         Grad/Evidenz/Abschnitt-Extraktion
│   │   └── reference.py        Bibliografie-Parsing
│   │
│   ├── ui/                     Streamlit-Frontend
│   │   ├── app.py              Einstiegspunkt
│   │   └── components/
│   │       ├── chat_page.py
│   │       ├── source_cards.py
│   │       ├── inline_citations.py
│   │       ├── insights_panels.py
│   │       ├── analytics_dashboard.py
│   │       └── filters.py
│   │
│   ├── telemetry.py            Token-Tracking, Kosten, Tool-Zusammenfassung
│   └── citations.py            Zitatformatierung
│
├── evaluations/                Evaluierungsframework
│   ├── scripts/
│   │   ├── run_eval.py         Testdatensatz gegen Live-API ausführen
│   │   └── run_ab_eval.py      A/B-Vergleich zweier Konfigurationen
│   ├── metrics/
│   │   ├── ragas_metrics.py    RAGAs-Integration
│   │   ├── retrieval.py        chunk_recall, chunk_precision
│   │   ├── behavioral.py       tool_call_count, external_search_used
│   │   └── similarity.py       answer_similarity, coverage
│   └── ui/
│       └── app.py              Evaluierungsergebnis-Dashboard
│
├── scripts/
│   ├── run_app.py              API + UI gemeinsam starten
│   ├── run_indexer.py          PDFs in Milvus indizieren
│   └── generate_eval_dataset.py
│
├── tests/                      Pytest-Testsuite
│   ├── api/
│   ├── graph/
│   ├── indexer/
│   ├── retrieval/
│   └── tools/
│
├── data/                       Laufzeitdaten (nicht versioniert)
│   └── app_state.db            SQLite-Gesprächsdatenbank
│
├── milvus.db/                  Milvus Lite lokal (nicht versioniert)
├── bm25_index.pkl              BM25 Sparse Index (nicht versioniert)
├── pyproject.toml
└── .env.example
```

---

## Tests ausführen

```bash
pytest                    # Alle Tests
pytest tests/retrieval/   # Einzelnes Modul
pytest -v                 # Mit ausführlicher Ausgabe
```

---

## Evaluierung

Evaluierungssuite gegen die laufende API ausführen:

```bash
python evaluations/scripts/run_eval.py        # Vollständige Evaluierung mit RAGAs-Metriken
python evaluations/scripts/run_ab_eval.py     # A/B-Vergleich zweier Modellkonfigurationen
streamlit run evaluations/ui/app.py           # Ergebnisse im Dashboard ansehen
```

Berechnete Metriken:
- **Retrieval:** `chunk_recall`, `chunk_precision`, `top_gold_chunk_hit`
- **RAGAs:** `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`
- **Behavioral:** `answer_length`, `tool_call_count`, `external_search_used`
- **Ähnlichkeit:** `answer_similarity`, `coverage`

Ergebnisse werden als `summary.json`, `item_results.json`, `ragas_records.json` und `metadata.json` gespeichert.

---

## Tech-Stack

| Komponente | Technologie |
|---|---|
| Graph-Orchestrierung | LangGraph |
| LLM — Generierung | GPT-4o via OpenRouter |
| LLM — günstige Aufgaben | Gemini 2.5 Flash via OpenRouter |
| Embeddings | OpenAI `text-embedding-3-large` (3072 Dim.) |
| Vektordatenbank | Milvus Lite (in-process, `./milvus.db`) |
| Sparse-Index | BM25 (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-v2-m3` (sentence-transformers) |
| PDF-Parsing | PyMuPDF |
| Backend-API | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (sse-starlette) |
| Frontend | Streamlit |
| Persistenz | SQLite (Gespräche) |
| Evaluierung | RAGAs |

---

## Security Layers

```
  INPUT                              OUTPUT
  ─────────────────────────────────  ────────────────────────────────────
  prompt injection → substring block patient-specific → regex → blocked
  off-topic        → keyword block   dosage w/o source → blocked
  PII              → regex redact    dosage w/ source  → allowed (grounded)
  auth             → API key check   hallucination     → NICHTS guard
  rate limit       → sliding window  training data     → GPT-4o never sees
                     per(route·key·ip)                   raw chunks
```

---
## Umgebungshinweise

- **Milvus Lite** läuft in-process — kein Docker oder externer Milvus-Server für lokale Entwicklung erforderlich. Die Datenbank liegt in `./milvus.db/`.
- **OpenRouter** stellt einen einheitlichen API-Endpunkt für GPT-4o und Gemini bereit. Nur ein API-Schlüssel wird benötigt.
- `MILVUS_URI` leer lassen (oder nicht setzen) für Milvus Lite. Eine HTTP-Adresse aktiviert einen vollständigen Milvus-Server.
- Für Produktion `DATABASE_URL` auf eine PostgreSQL-Verbindungszeichenkette setzen, um LangGraph-Graph-Checkpointing über Server-Neustarts hinweg zu aktivieren.


---

## Key Abbreviations & Concepts



| Term | Meaning in this project |
|---|---|
| **Leitlinie** | German word for clinical guideline. S3 is the highest evidence level (systematic evidence review). |
| **Empfehlung** | Recommendation — a specific clinical action statement inside a Leitlinie, e.g. "Empfehlung 4.2.1". |
| **Empfehlungsgrad** | Recommendation grade: A (strong), B (moderate), 0 (open). |
| **Evidenzlevel** | Evidence level: 1a/1b (RCTs), 2a/2b (cohort studies), 3/4/5 (lower evidence). |
| **Leaf chunk** | A chunk with no children — the actual content chunk that gets embedded and searched. Parent chunks are sections that contain leaf chunks. |
| **Dense vector / embedding** | A 3072-dimensional float array representing the semantic meaning of a text. Computed by OpenAI's `text-embedding-3-large`. Semantically similar texts have similar vectors. |
| **ANN (Approximate Nearest Neighbor)** | Algorithm to find the closest vectors to a query vector without comparing to every vector. Milvus uses HNSW for this. |
| **HNSW** | Hierarchical Navigable Small World. A graph-based ANN index that allows very fast approximate nearest-neighbor search on high-dimensional vectors. Parameters: M=16 (connections per node), efConstruction=200 (build-time quality). |
| **BM25** | Best Match 25. A classical keyword-based ranking function. Unlike dense search, it ranks documents by exact term frequency and inverse document frequency. Good for exact drug names, recommendation IDs. BM25Okapi is a popular variant with parameter tuning. |
| **Sparse retrieval** | Search based on keyword matching (BM25). Returns only chunks that contain the exact query terms. |
| **Dense retrieval** | Search based on embedding similarity (Milvus). Returns chunks that are semantically similar even if they use different words. |
| **RRF (Reciprocal Rank Fusion)** | A score fusion formula: `score = Σ 1/(k + rank + 1)` for each ranked list. Combines multiple ranked lists into one without needing calibrated scores. k=60 is the standard smoothing constant. |
| **Hybrid search** | Combination of dense + sparse retrieval fused via RRF. Gets the best of both worlds: semantic matching from dense, exact term matching from sparse. |
| **Cross-encoder / Reranker** | A model (`BAAI/bge-reranker-v2-m3`) that takes a (query, passage) pair as input and outputs a relevance score. More accurate than embedding similarity but slower — used after an initial fast retrieval to re-sort the top candidates. |
| **Parent expansion** | Each leaf chunk has a `parent_chunk_id`. Before returning results, the system fetches the parent chunk text and prepends it to the leaf text. This gives the LLM more context about where the sentence comes from.|
| **TypedDict** | Python type hint for dict with known keys and typed values. `RAGState` is a TypedDict containing all fields that flow through the graph. |
| **OpenRouter** | API proxy that routes calls to multiple LLM providers (OpenAI, Google, Anthropic) using a single API key and unified OpenAI-compatible interface. |
| **Gemini 2.5 Flash** | Google's fast, cheap model. Used here for operations that need to be quick and cost-efficient: query rewriting, turn classification, verbatim extraction. |
| **GPT-4o** | OpenAI's high-capability model. Used here for: running the tool-calling agent loop, synthesizing the final answer, memory-based rewrites. |
| **SSE (Server-Sent Events)** | HTTP protocol for server-to-client streaming. Used by the FastAPI backend to stream partial answers to the Streamlit UI as they are generated. |
| **RBAC** | Role-Based Access Control. Users have a `user_role` in state (user/professional/admin) and this determines which tools they can call. |
| **PII** | Personally Identifiable Information. Patient names, birthdates, phone numbers, postal codes. The input guardrail redacts these before the query is processed further. |

