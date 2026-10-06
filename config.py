"""Central configuration for the Banking Agent project.

Model choice (per project requirements):
  - Agent LLM : OpenAI GPT models via the OpenAI API (gap-gpt key works too —
                just set OPENAI_BASE_URL). gpt-4o is used because it is one of
                the strongest models for Persian (Farsi) understanding,
                function-calling and reasoning.
  - Embeddings: a dedicated Persian sentence-transformers model
                ("Msobhi/Persian_Sentence_Embedding_v3", fine-tuned on the
                Persian MIRACL retrieval dataset; alternatives:
                "myrkur/sentence-transformer-parsbert-fa-2.0"). It runs locally
                & offline so no extra API cost is incurred, while giving strong
                Persian retrieval quality for the RAG pipeline.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CUSTOMERS_FILE = os.environ.get(
    "CUSTOMERS_FILE", os.path.join(DATA_DIR, "customers.xlsx")
)
CARD_ORDERS_FILE = os.environ.get(
    "CARD_ORDERS_FILE", os.path.join(DATA_DIR, "card_orders.xlsx")
)
POLICIES_FILE = os.environ.get(
    "POLICIES_FILE",
    os.path.join(DATA_DIR, "قوانین و مقررات بانک دایا_تدبیر.docx"),
)
REQUESTS_FILE = os.environ.get(
    "REQUESTS_FILE", os.path.join(BASE_DIR, "requests.xlsx")
)

# ---------------------------------------------------------------------------
# OpenAI (agent LLM + optional embeddings)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or None  # custom gateway support

# The agent model must be excellent at Persian -> gpt-4o (or gpt-4.1).
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# "openai"  -> text-embedding-3-small/large (multilingual, good Persian)
# "persian" -> csebuetnlp/mMTEB_large_fa   (best dedicated Persian model, local)
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "persian").lower()
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PERSIAN_EMBEDDING_MODEL = os.environ.get(
    "PERSIAN_EMBEDDING_MODEL", "csebuetnlp/mMTEB_large_fa"
)

# ---------------------------------------------------------------------------
# Retrieval / behaviour knobs
# ---------------------------------------------------------------------------
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# Card issuance standard working-day windows from the policy document
# (used for temporal reasoning: order date + max working days vs today).
CARD_ISSUANCE_DAYS = {
    "اعتباری طلایی": (3, 4),
    "نقدی طلایی": (3, 5),
    "هدیه": (3, 5),
    "اعتباری نقره‌ای": (4, 6),
    "اعتباری": (4, 6),      # generic fallbacks
    "نقدی": (5, 7),
    "استاندارد": (5, 7),
}
DEFAULT_ISSUANCE_WINDOW = (3, 7)  # general bank policy statement

ACTIVE_STATUSES = [
    "در دست بررسی",
    "تأیید شده",
    "در حال چاپ",
    "ارسال به پست",
    "ارسال شده",
]
DELIVERED_STATUS = "تحویل داده شده"

REQUEST_TYPES = ["وام", "افتتاح حساب", "سفارش کارت جدید", "شکایت", "استعلام", "سایر"]

INTENTS = [
    "card_status",        # پیگیری وضعیت کارت
    "new_card_request",   # درخواست کارت جدید
    "loan_eligibility",   # شرایط/استحقاق وام
    "policy_question",    # سؤال عمومی از قوانین (RAG)
    "request_logging",    # ثبت درخواست (وام/حساب/شکایت/...)
    "customer_info",      # اطلاعات مشتری (موجودی، امتیاز و...)
    "greeting",           # سلام و احوالپرسی
    "clarify",            # جمله مبهم بدون context
    "other",
]
