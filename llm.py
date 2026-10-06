"""LLM factory + structured-output helpers built on langchain-openai.

The agent brain is an OpenAI GPT model (gpt-4o by default — excellent at
Persian). The API key/base-url come from the environment so any OpenAI-compatible
gateway (e.g. gap-gpt) works out of the box.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config

log = logging.getLogger("banking_agent.llm")


class IntentResult(BaseModel):
    """LLM-based intent classification (Constraint 1 — no keyword matching)."""

    intent: str = Field(
        description=(
            "Exactly one of: card_status | new_card_request | loan_eligibility | "
            "policy_question | request_logging | customer_info | greeting | clarify | other"
        )
    )
    national_code: str | None = Field(default=None, description="10-digit Iranian national code if mentioned")
    card_type: str | None = Field(default=None, description="اعتباری/نقدی/هدیه if mentioned")
    request_type: str | None = Field(default=None, description="وام/افتتاح حساب/سفارش کارت جدید/شکایت/استعلام/سایر if it is a request")
    mentions_loan: bool = Field(default=False)
    mentions_complaint: bool = Field(default=False)
    needs_clarification: bool = Field(default=False, description="true if the message is ambiguous and context is missing")
    reasoning_note: str = Field(default="", description="one short Persian sentence explaining the choice")


def make_llm(model: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for Persian banking dialogue."""
    if not config.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put your OpenAI/gap-gpt key in .env "
            "(OPENAI_API_KEY=sk-...) or export it before running the agent."
        )
    llm = ChatOpenAI(
        model=model or config.AGENT_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        timeout=90,
        max_retries=2,
    )
    log.info("LLM ready: %s (base_url=%s)", model or config.AGENT_MODEL,
             config.OPENAI_BASE_URL or "https://api.openai.com/v1")
    return llm


def classify_intent(llm: ChatOpenAI, message: str, context_summary: str) -> dict:
    """Classify user intent with the LLM (structured output)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    sys = (
        "You are the intent-classification module of a Persian banking assistant.\n"
        "Read the user's latest Persian message plus the conversation context and "
        "classify its intent. Do NOT rely on literal keywords — understand meaning "
        "(e.g. «کارتم نیومد هنوز» → card_status; «یه کارت دیگه میخوام» → new_card_request; "
        "«کارتم گم شد» → request_logging/complaint; «چند وقت طول میکشه کارت بیاد؟» → policy_question).\n"
        "If the message contains pronouns like «وضعیتش», «اون», «همون» without enough "
        "context to resolve the referent, set needs_clarification=true and intent='clarify'.\n"
        "Extract any 10-digit national code exactly as written (keep leading zero)."
    )
    human = f"CONTEXT:\n{context_summary or '(none)'}\n\nUSER MESSAGE:\n{message}"
    try:
        st = llm.with_structured_output(IntentResult)
        res = st.invoke([SystemMessage(content=sys), HumanMessage(content=human)])
        return res.model_dump()
    except Exception as exc:
        log.warning("intent classification failed (%s); falling back to 'other'", exc)
        return {
            "intent": "other", "national_code": None, "card_type": None,
            "request_type": None, "mentions_loan": False, "mentions_complaint": False,
            "needs_clarification": False, "reasoning_note": f"خطا: {exc}",
        }
