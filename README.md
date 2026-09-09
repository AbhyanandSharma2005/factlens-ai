# FactLens AI — A Fact Knowledge Layer for PDFs

Extracts grounded, evidence-linked facts from PDFs and automatically determines
whether facts across documents **corroborate**, **contradict**, or are
**reconciled by context** (different time periods, units, or methodology).

---

## Table of Contents

- [Live Demo](#live-demo)
- [Setup and Run Instructions](#setup-and-run-instructions)
- [Video Demo](#video-demo)
- [Architecture](#architecture)
- [Approach](#approach)
- [The Four Required Cases](#the-four-required-cases)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Additional Notes](#additional-notes)

---

## Live Demo

🖥️ **App:** [https://factlens-ai.vercel.app/](https://factlens-ai.vercel.app/)
🔌 **API docs:** [https://factlens-ai-km45.onrender.com/docs](https://factlens-ai-km45.onrender.com/docs)

> Frontend is hosted free on Vercel (static, no cold starts). Backend is
> hosted free on Render — the instance sleeps after 15 minutes of
> inactivity, so the **first** request after a period of no traffic may take
> 30–60 seconds to respond while it wakes up. If the app looks like it's
> hanging on first load, that's why — give it a minute.

---

## Setup and Run Instructions

### Prerequisites
- Python 3.12+
- A Postgres database with the [`pgvector`](https://github.com/pgvector/pgvector) extension enabled (a free [Neon](https://neon.tech) instance works out of the box)
- A [Groq API key](https://console.groq.com/keys) (free tier)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/factlens-ai.git
cd factlens-ai/backend
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

`requirements.txt` is version-pinned to a tested working set — if you hit
a build error on a fresh machine or on Render, check that no dependency
has since introduced a breaking change against these exact pins.

### 2. Configure environment

Create `backend/.env`:

```
GROQ_API_KEY="your-groq-key-here"
GROQ_MODEL=qwen/qwen3.6-27b
DATABASE_URL="postgresql://user:pass@host/dbname?sslmode=require"
```

Never commit `.env`. Confirm it's listed in `.gitignore` before pushing.

### 3. Create the schema

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

CREATE INDEX ON facts (document_id);
CREATE INDEX ON facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### 5. Upload a PDF

```bash
curl -X POST "http://127.0.0.1:8000/api/documents" \
  -F "file=@your-document.pdf"
```

Optional `?max_pages=N` caps ingestion for cheap iterative testing without
burning API quota on a full run.

### 6. Inspect results

```bash
curl http://127.0.0.1:8000/api/facts
```

Returns every extracted fact with its source evidence, page number, and any
cross-document relationships discovered.

### 7. Run the frontend (optional, separate app)

```bash
cd ../frontend
npm install
```

Create `frontend/.env`:

```
VITE_API_URL=http://127.0.0.1:8000
```

```bash
npm run dev
```

---

## Video Demo

📺 [Watch the demo](#)

Shows a PDF being uploaded, processed live, and walks through all four
required cases with source evidence on screen.

---

## Architecture

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

    subgraph Pipeline["Ingestion Pipeline (per page)"]
        Extract["extraction.py\nper-page LLM extraction"]
        Guard["Hallucination Guard\nverbatim substring check"]
        Embed["fastembed\nBAAI/bge-small-en-v1.5"]
        Recon["reconciler.py\ncross-doc adjudication"]
    end

    Groq[("Groq API\nqwen/qwen3.6-27b")]
    DB[("Postgres + pgvector\n(Neon)")]

    User -->|"upload PDF"| FE
    FE -->|"axios"| Upload
    Upload -->|"enqueue"| BG
    BG --> Extract
    Extract <-->|"extraction prompt"| Groq
    Extract --> Guard
    Guard -->|"verified facts"| Embed
    Embed -->|"store fact + vector"| DB
    Embed -->|"vector similarity\nsearch < 0.28 cosine"| DB
    DB -->|"candidate facts\nfrom other documents"| Recon
    Recon <-->|"adjudication prompt"| Groq
    Recon -->|"store relationship"| DB
    FE -->|"axios"| Facts
    Facts --> DB
```

### Per-Page Extraction Flow

```mermaid
flowchart TD
    A["PyMuPDF: extract page text"] --> B{"< 100 chars OR\nlooks_like_toc_or_cover?"}
    B -->|"yes"| Skip["Skip — no LLM call"]
    B -->|"no"| C["LLM extraction call\n(Groq, JSON mode, temp=0)"]
    C --> D{"Retryable error?"}
    D -->|"429 / 5xx"| E["Exponential backoff, retry"]
    E --> C
    D -->|"400 / schema error"| F["Fail fast — no retry"]
    C --> G["Parse facts individually\n(one bad fact ≠ page discarded)"]
    G --> H{"evidence_text is exact\nverbatim substring of page?"}
    H -->|"no"| Reject1["❌ REJECTED — hallucination"]
    H -->|"yes"| I{"Contains a number\n/ date / statistic?"}
    I -->|"no"| Reject2["❌ REJECTED — no quantifiable content"]
    I -->|"yes"| J["Embed + locate bounding box\n→ store as verified fact"]
```

### Cross-Document Reconciliation

```mermaid
flowchart LR
    NF["New fact inserted"] --> VS["Vector similarity search\n(other documents only,\ncosine distance < 0.28)"]
    VS --> Dup{"Near-identical text?\n(is_near_identical)"}
    Dup -->|"yes — structural dup\n(e.g. TOC vs heading)"| Skip["Skip — no LLM call"]
    Dup -->|"no"| LLM["Adjudicate via LLM"]
    LLM --> R1["corroborates"]
    LLM --> R2["contradicts"]
    LLM --> R3["reconciled_by_context"]
    LLM --> R4["insufficient_evidence"]
    R1 --> Store["Store in fact_relationships"]
    R2 --> Store
    R3 --> Store
    R4 --> Discard["Not stored"]
```

### Why this design

- **Verbatim evidence grounding** — every fact must be an exact substring of
  the source page text, so the system cannot silently hallucinate a number.
- **Embedding-gated reconciliation** — pairwise LLM adjudication only runs on
  facts that are already semantically close (cosine distance < 0.28), which
  keeps the system's cost roughly linear in fact count rather than quadratic,
  and naturally clusters comparisons by topic without any hard-coded schema.
- **Schema-agnostic facts** — `subject` / `fact_type` / `metric_value` /
  `scope_context` are free-text and LLM-populated, so the system was never
  told what a "fact" looks like in these specific documents; it generalizes
  to any PDF.

---

## Approach

1. **Extraction** — each PDF page is sent to Groq with a prompt requiring
   3–5 verifiable, numeric claims per page and an exact verbatim quote as
   evidence. A cheap structural heuristic (`looks_like_toc_or_cover`) skips
   TOC/cover pages before spending an LLM call on them.
2. **Grounding** — every returned fact is checked against the raw page text
   for an exact (unicode-normalized) substring match. Facts that fail are
   logged and dropped as hallucinations — never silently kept.
3. **Embedding** — verified facts are embedded (`BAAI/bge-small-en-v1.5`,
   384-dim) and stored alongside their evidence page, verbatim quote, and a
   bounding box for UI highlighting.
4. **Reconciliation** — on insert, each fact is compared via vector search
   against facts from other documents. Candidates within 0.28 cosine
   distance are adjudicated by a second LLM call into one of four relation
   types, with the explanation stored alongside the relationship.
5. **Resilience** — a shared retryable-error classifier distinguishes
   transient failures (rate limits, 5xx) from deterministic ones (400s,
   schema failures), so the pipeline fails fast on the latter instead of
   wasting minutes retrying a request that will never succeed.

**AI tools used:** Claude (Anthropic) for architecture review, debugging, and
code fixes throughout development; Groq-hosted open models for the extraction
and reconciliation LLM calls in the running system itself.

---

## The Four Required Cases

| # | Case | Example | Evidence |
|---|------|---------|----------|
| 1 | **Corroborates** | FY24 EBITDA reported as ₹1,266Mn in the Annual Report vs. Rs. 127 Cr in the Q4 FY24 Earnings Presentation — same figure, rounded differently, two independent documents. | Annual Report p.4; Earnings Presentation p.5 |
| 2 | **Contradicts** | Real GDP Growth projection for FY25: India Economic Survey forecasts 6.5–7.0%, while the IMF Article IV report forecasts 5.8%. The system correctly identified this as a direct numerical conflict for the exact same indicator, entity, and timeframe. | Economic Survey p.14; IMF Article IV p.28 |
| 3 | **Reconciled by context** | ₹2,076 Cr Q4 FY24 revenue vs. ₹8,141.5 Cr full-year FY24 revenue — correctly explained as a quarter vs. full-year comparison, not a conflict. | Earnings Presentation p.7; Annual Report p.6 |
| 4 | **Extraction/reasoning failure** | A dense two-column financial table (Mar'23 / Mar'24) had its evidence text captured correctly (passing the hallucination guard) but only the first column's value was structured into `metric_value` — silently dropping the second year as a separate fact. | Earnings Presentation p.20 |

**Case 2 (Contradicts) Adjudication Output:**

```json
{
  "relation_type": "contradicts",
  "explanation": "Both facts project real GDP growth for India in FY25. Fact A forecasts 6.5–7.0%, while Fact B forecasts 5.8%. Since they refer to the exact same economic indicator, entity, and fiscal year without any stated methodological, temporal, or unit differences to explain the gap, the non-overlapping numerical projections constitute a direct conflict.",
  "confidence": 0.95
}
```

---

## Limitations and Next Steps

### Known limitations

- **Table-structure loss** — PyMuPDF's flat text extraction can reorder
  multi-column tables, occasionally producing evidence text that reads
  correctly but caused false hallucination rejections before a unicode
  normalization fix; residual cases remain on dense tables (see Case 4).
- **TOC/cover heuristic false positives** — `looks_like_toc_or_cover` can
  skip narrative-prose pages that legitimately contain no numeric facts but
  are structurally similar to a skippable page; not independently verified
  against every page in this run.
- **Free-tier API limits** — Groq's free tier caps both requests-per-minute
  and tokens-per-day; large documents were tested with an optional
  `max_pages` cap during development to conserve quota, with full,
  uncapped runs reserved for the final demo.
- **Single embedding model, no reranking** — reconciliation candidates are
  gated purely by cosine distance; a borderline-relevant pair just outside
  the 0.28 threshold is never checked, and an irrelevant pair just inside it
  costs an LLM call that returns `insufficient_evidence`.

### Next steps

- Table-aware extraction (e.g. `pdfplumber`'s `extract_table()`) for
  pages with dense multi-column data, to fully resolve Case 4's root cause.
- Structured multi-value facts (e.g. one fact per table row/column pair)
  instead of collapsing an entire row into a single `metric_value`.
- A confidence-weighted secondary rerank on reconciliation candidates
  instead of a hard cosine cutoff.
- Incremental re-ingestion: currently, re-uploading a corrected/updated PDF
  creates a new `document_id` rather than updating in place.

---

## Additional Notes

- All fact and relationship data is fully schema-agnostic — `fact_type`,
  `subject`, and `scope_context` are LLM-populated free text, so the system
  was tested successfully across two unrelated document domains (India
  macroeconomic reports and corporate filings) in the same
  knowledge layer without any code changes.
- No secrets are committed to this repository. `.env.example` is provided
  as a template; the actual `.env` is gitignored.
- Deployment stack: frontend on Vercel (free, static, no cold starts);
  backend on Render (free web service, sleeps after 15 min idle); database
  on Neon (free Postgres with pgvector). All three tiers are genuinely
  free with no credit card required.