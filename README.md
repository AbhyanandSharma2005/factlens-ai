# FactLens AI — A Fact Knowledge Layer for PDFs

Extracts grounded, evidence-linked facts from PDFs and automatically determines whether facts across documents **corroborate**, **contradict**, or are **reconciled by context** (different time periods, units, or methodology).

---

## Table of Contents

* [Live Demo](https://www.google.com/search?q=%23live-demo)
* [Setup and Run Instructions](https://www.google.com/search?q=%23setup-and-run-instructions)
* [Video Demo](https://www.google.com/search?q=%23video-demo)
* [Architecture & Resilient Design](https://www.google.com/search?q=%23architecture--resilient-design)
* [The Four Required Cases](https://www.google.com/search?q=%23the-four-required-cases)
* [Limitations and Next Steps](https://www.google.com/search?q=%23limitations-and-next-steps)
* [Additional Notes](https://www.google.com/search?q=%23additional-notes)

---

## Live Demo

🖥️ **App:** [https://factlens-ai.vercel.app/](https://factlens-ai.vercel.app/)
🔌 **API docs:** [https://factlens-ai-km45.onrender.com/docs](https://factlens-ai-km45.onrender.com/docs)

> Frontend is hosted free on Vercel (static, no cold starts). Backend is hosted free on Render — the instance sleeps after 15 minutes of inactivity, so the **first** request after a period of no traffic may take 30–60 seconds to respond while it wakes up. If the app looks like it's hanging on first load, that's why — give it a minute.

---

## Setup and Run Instructions

### Prerequisites

* Python 3.12+
* A Postgres database with the [`pgvector`](https://github.com/pgvector/pgvector) extension enabled (a free [Neon](https://neon.tech) instance works out of the box)
* A [Groq API key](https://console.groq.com/keys) (free tier)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/factlens-ai.git
cd factlens-ai/backend
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

```

`requirements.txt` is version-pinned to a tested working set — if you hit a build error on a fresh machine or on Render, check that no dependency has since introduced a breaking change against these exact pins.

### 2. Configure environment

Create `backend/.env`:

```env
GROQ_API_KEY="your-groq-key-here"
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_URL="postgresql://user:pass@host/dbname?sslmode=require"

```

Never commit `.env`. Confirm it's listed in `.gitignore` before pushing.

### 3. Create the schema

Run the following SQL script in your Neon Postgres SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'processing',
    total_pages INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    metric_value JSONB NOT NULL,
    scope_context JSONB DEFAULT '{}',
    qualifiers TEXT[] DEFAULT '{}',
    evidence_page INT NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_bbox JSONB,
    confidence FLOAT DEFAULT 0.5,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE fact_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_a_id UUID REFERENCES facts(id) ON DELETE CASCADE,
    fact_b_id UUID REFERENCES facts(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    explanation TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    UNIQUE (fact_a_id, fact_b_id)
);

-- Content-addressable caching layer for zero-token repeat extractions
CREATE TABLE IF NOT EXISTS fact_cache (
    text_hash TEXT PRIMARY KEY,
    cached_facts JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON facts (document_id);
CREATE INDEX ON facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

```

### 4. Run

```bash
uvicorn app.main:app --reload

```

Open `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)` for the interactive Swagger UI.

### 5. Upload a PDF

```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -F "file=@your-document.pdf"

```

### 6. Inspect results

```bash
curl http://127.0.0.1:8000/api/facts

```

### 7. Run the frontend (optional, separate app)

```bash
cd ../frontend
npm install

```

Create `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000

```

```bash
npm run dev

```

---

## Video Demo

📺 [Watch the demo](https://www.google.com/search?q=%23)

Shows a PDF being uploaded, processed live, and walks through all four required cases with source evidence on screen.

---

## Architecture & Resilient Design

### System Overview

```mermaid
flowchart LR
    User([Browser])
    FE["React + Vite Frontend\n(Vercel)"]

    subgraph API["FastAPI Backend (Render)"]
        Upload["POST /api/documents"]
        Facts["GET /api/facts"]
        BG["Background Task Runner"]
    end

    subgraph Pipeline["Optimized Ingestion Pipeline (per page)"]
        PreFilter["Local Regex Block Filter\n(Discards zero-digit text)"]
        Cache{"Cache Hit?\n(SHA-256 Hash)"}
        Pacer["Token-Bucket Pacer\n(Self-regulating frequency)"]
        Extract["extraction.py\nLLM extraction"]
        Defensive["Defensive JSON Parser\n(Regex markdown recovery)"]
        Guard["Hallucination Guard\nverbatim substring check"]
        Embed["fastembed\nBAAI/bge-small-en-v1.5"]
    end

    Groq[("Groq API\nInference Model")]
    DB[("Postgres + pgvector\n(Neon with fact_cache)")]

    User -->|"upload PDF"| FE
    FE -->|"axios"| Upload
    Upload -->|"enqueue"| BG
    BG --> PreFilter
    PreFilter --> Cache
    Cache -->|"yes (0 tokens)"| DB
    Cache -->|"no"| Pacer
    Pacer --> Extract
    Extract <-->|"call API"| Groq
    Extract --> Defensive
    Defensive --> Guard
    Guard -->|"verified facts"| Embed
    Embed -->|"store fact + vector"| DB
    Embed -->|"save to cache"| DB

```

### Key Architectural Optimizations

1. **Heuristic Block-Level Pre-Filtering:** Before making any external API call, PyMuPDF extracts text by paragraphs (blocks). A local Python regex filter instantly discards any text block containing zero digits or numbers, cutting input token consumption by up to 80%.
2. **Content-Addressable Caching (`fact_cache`):** Every filtered text chunk is cryptographically hashed using SHA-256. The system checks Neon Postgres before hitting the LLM; if a page or text segment has been processed previously, cached JSON facts are returned instantly for **zero tokens and zero latency**.
3. **Proactive Token-Bucket Pacer:** Instead of relying on reactive error handling for rate limits, a thread-safe token-bucket rate limiter mathematically self-regulates outbound request frequency to stay well within provider thresholds.
4. **Defensive JSON Recovery Layer:** To eliminate strict API-level JSON validation failures (`400 Bad Request`), the pipeline utilizes a robust regex-based extraction parser. It successfully isolates and parses JSON payloads even if models wrap them in markdown code blocks or conversational text.
5. **Verbatim Evidence Grounding:** Every fact must be an exact substring of the source page text, preventing silent hallucinations.

---

## The Four Required Cases

| # | Case | Example | Evidence |
| --- | --- | --- | --- |
| 1 | **Corroborates** | FY24 EBITDA reported as ₹1,266Mn in the Annual Report vs. Rs. 127 Cr in the Q4 FY24 Earnings Presentation — same figure, rounded differently, two independent documents. | Annual Report p.4; Earnings Presentation p.5 |
| 2 | **Contradicts** | Real GDP Growth projection for FY25: India Economic Survey forecasts 6.5–7.0%, while the IMF Article IV report forecasts 5.8%. The system correctly identified this as a direct numerical conflict for the exact same indicator, entity, and timeframe. | Economic Survey p.14; IMF Article IV p.28 |
| 3 | **Reconciled by context** | ₹2,076 Cr Q4 FY24 revenue vs. ₹8,141.5 Cr full-year FY24 revenue — correctly explained as a quarter vs. full-year comparison, not a conflict. | Earnings Presentation p.7; Annual Report p.6 |
| 4 | **Extraction/reasoning failure** | A dense two-column financial table (Mar'23 / Mar'24) had its evidence text captured correctly (passing the hallucination guard) but only the first column's value was structured into `metric_value` — silently dropping the second year as a separate fact. | Earnings Presentation p.20 |

---

## Limitations and Next Steps

### Known limitations

* **Table-structure loss** — PyMuPDF's flat text extraction can reorder multi-column tables, occasionally producing evidence text that causes false hallucination rejections or drops separate table columns.
* **Free-tier API limits** — Groq's free tier caps daily token totals; large documents can trigger daily caps without proper chunking or API key rotation.
* **Single embedding model, no reranking** — reconciliation candidates are gated purely by cosine distance.

### Next steps

* Table-aware extraction (e.g. `pdfplumber`) for pages with dense multi-column data.
* Structured multi-value facts instead of collapsing rows into a single `metric_value`.
* Incremental re-ingestion pipelines.

---

## Additional Notes

* All fact and relationship data is fully schema-agnostic — `fact_type`, `subject`, and `scope_context` are LLM-populated free text.
* No secrets are committed to this repository. `.env` files are gitignored.
* Deployment stack: frontend on Vercel; backend on Render; database on Neon with `pgvector` and `fact_cache`.