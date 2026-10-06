# 🏦 Persian Banking Agent (دستیار هوشمند بانک دایا‌تدبیر)

LangGraph-powered multi-step banking agent in **Persian**, using:

- **Agent LLM:** OpenAI `gpt-4o` via the OpenAI-compatible API — works directly with your **gap-gpt key** (set `OPENAI_BASE_URL` to the gateway). gpt-4o is one of the strongest models for Persian understanding, function-calling and reasoning.
- **Embeddings (RAG):** best-of-class *local* Persian model `csebuetnlp/mMTEB_large_fa` (sentence-transformers — free, offline, top MTEB-Fa retrieval quality). Switchable to OpenAI `text-embedding-3-small` via `EMBEDDING_PROVIDER=openai`.
- **Retrieval:** hybrid dense-vector + BM25 (EnsembleRetriever) over the bank policy `.docx`.
- **Tools:** customer lookup, card-order tracking (Jalali working-day math), request logging to Excel — all on LangChain v1 / LangGraph v1 APIs.

## Files

| File | Purpose |
|---|---|
| `main.py` | CLI chat / demo / single-shot question |
| `app_ui.py` | Streamlit web UI |
| `agent.py` | LangGraph state machine (intent → tools → reasoning → answer) |
| `llm.py` | ChatOpenAI factory + structured intent classification |
| `rag.py` | Policy RAG: docx loading, chunking, Persian embeddings, hybrid retrieval |
| `tools.py` | Agent tools (card status, new card, loan eligibility, requests…) |
| `data_access.py` | pandas layer over `customers.xlsx` / `card_orders.xlsx` |
| `jalali_utils.py` | Jalali ⇄ Gregorian conversion, working-day counting |
| `config.py` | All knobs (model names, paths, thresholds) |

## Requirements

Python **3.10+**. Install dependencies:

```bash
pip install -r requirements.txt
```

## 1) Configure the API key

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Your gap-gpt key:
OPENAI_API_KEY=sk-your-gapgpt-key-here
OPENAI_BASE_URL=https://api.gap-gpt.com/v1   # ← only if you use the gap-gpt gateway;
                                             #   leave unset for direct api.openai.com
AGENT_MODEL=gpt-4o                           # must be a model your key can access
EMBEDDING_PROVIDER=persian                   # local Persian model (default, free)
```

> On first run the embedding model (~2 GB) is downloaded from Hugging Face automatically
> and cached in `~/.cache/huggingface`. After that everything works offline except LLM calls.
> To use OpenAI embeddings instead, set `EMBEDDING_PROVIDER=openai`.

## 2) Run it

### Interactive CLI chat (multi-turn, keeps context)
```bash
python main.py
```

### Scripted multi-scenario demo (good for testing/grading)
```bash
python main.py --demo
```

### Single-shot question
```bash
python main.py -q "سلام، کد ملی من 0221536568 هست. کارتم چی شد؟"
```

Useful flags: `--no-reasoning` (hide the step-by-step reasoning trace), `--thread ID` (resume a conversation thread).

### Streamlit web UI
```bash
streamlit run app_ui.py
```
Then open http://localhost:8501 (RTL Persian interface).

## Example conversation

```
🧑 کاربر: سلام، کد ملی من 0221536568 هست. کارتم چی شد؟
🤖 دستیار: وضعیت کارت شما «در حال چاپ» است … (with Jalali working-day reasoning)

🧑 کاربر: میخوام یه کارت اعتباری جدید هم بگیرم
🤖 دستیار: برای ثبت درخواست کارت جدید … (writes to requests.xlsx)
```

## Troubleshooting

- `OPENAI_API_KEY is not set` → create `.env` (see step 1) or `export OPENAI_API_KEY=...`.
- `401/404` from the gateway → wrong `OPENAI_BASE_URL`, or `AGENT_MODEL` not available on your key (try `gpt-4o-mini`).
- Embedding download slow/blocked → set `HF_ENDPOINT=https://hf-mirror.com`, or switch to `EMBEDDING_PROVIDER=openai`. The agent also auto-falls back to a lightweight hashing embedder so it never hard-crashes.
