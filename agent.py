"""The Banking Agent — LangGraph orchestration (latest langgraph 1.x API).

Graph:

    START → classify_intent ─┬─→ prepare_context ─┬─→ agent ⇄ tools (ReAct loop)
                             │                    │
                             └─→ clarify ─────────┘  (asks the user back)

* classify_intent : LLM structured-output intent classification (Constraint 1,
  no keyword matching). Also injects a deterministic hint when the conversation
  context already knows the authenticated customer.
* prepare_context : builds the Persian system prompt containing today's Jalali
  date, the identified customer, last topic and card-order summary so the agent
  can resolve ambiguous follow-ups (Constraint 3) and reason temporally
  (Constraint 6).
* agent           : ChatOpenAI(gpt-4o) with bound tools; multi-step reasoning
  visible through `reasoning_log` (Constraint 2).
* tools           : ToolNode running tools.py (auth, orders, RAG, hybrid loan,
  request logging).

State is persisted per-thread with InMemorySaver checkpointer → real
multi-turn context management (bonus requirement).
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

import config
from data_access import normalize_national_code
from llm import classify_intent, make_llm
from tools import get_tools

log = logging.getLogger("banking_agent.graph")

SYSTEM_CORE = (
    "تو دستیار هوشمند بانک «دایا‌تدبیر» هستی. همیشه به فارسی روان و رسمی پاسخ بده.\n"
    "قوانین رفتاری:\n"
    "۱) برای هر کار عملی اول با ابزار get_customer کاربر را با کد ملی احراز کن؛ اگر کد ملی در "
    "متن نیست از کاربر بخواه. اگر کد ملی نامعتبر یا ناموجود بود، مودبانه خطا بده.\n"
    "۲) برای وضعیت کارت از list_card_orders استفاده کن. اگر چند سفارش فعال داشت، همه را فهرست "
    "کن و بپرس کدام مدنظر است. اگر سفارشی نبود، پیشنهاد ثبت سفارش جدید بده. اگر آخرین سفارش "
    "«تحویل داده شده» بود، بپرس آیا کارت را دریافت نکرده است؟\n"
    "۳) برای سؤالات قوانین حتماً از search_policies استفاده کن و فقط بر اساس متن بازیابی‌شده "
    "پاسخ بده؛ عدد/کارمزد/روز کاری را دقیق از متن نقل کن.\n"
    "۴) سؤال ترکیبی (مثل «آیا می‌تونم وام بگیرم؟») را با check_loan_eligibility و در صورت نیاز "
    "search_policies پاسخ بده و نتیجه را با امتیاز/موجودی مشتری توضیح بده.\n"
    "۵) درخواست‌های کاربر (وام، افتتاح حساب، سفارش کارت جدید، شکایت، استعلام) را با log_request "
    "ثبت کن. قبل از سفارش کارت جدید، وجود سفارش فعال را با list_card_orders(only_active=true) "
    "چک کن. اعتبارسنجی تکراری بودن را ابزار انجام می‌دهد؛ پیامش را منتقل کن.\n"
    "۶) استدلال چندمرحله‌ای: قبل از فراخوانی ابزارها در یک پاراگراف کوتاه فارسی («🧠 مراحل تفکر»)\n"
    "بنویس چه می‌کنی و چرا؛ مثلاً محاسبه فاصله تاریخ سفارش تا امروز (تقویم جلالی) و مقایسه با "
    "بازه استاندارد صدور از قوانین. اگر سفارش تأخیر دارد، تماس با پشتیبانی ۰۲۱-۳۴۵۶۷۸9 را پیشنهاد بده.\n"
    "۷) ابهام: اگر کاربر با ضمیر مبهم («وضعیتش چیه؟») صحبت کرد و موضوع قبلی در context مشخص است، "
    "همان را ادامه بده؛ اگر مشخص نیست، سؤال رفع ابهام کن.\n"
    "۸) تاریخ‌ها همیشه جلالی (مثلاً ۱۴۰۴/۱۰/۰۵) و مبالغ به تومان گزارش شوند.\n"
    "۹) محرمانگی: اطلاعات مشتری را فقط به همان کاربر نشان بده و شماره تماس کامل را فاش نکن."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    national_code: str | None
    card_type: str | None
    request_type: str | None
    needs_clarification: bool
    topic: str            # last discussion topic for context resolution
    reasoning_log: list   # observable multi-step reasoning trace


def _context_summary(state: AgentState) -> str:
    parts = []
    if state.get("national_code"):
        parts.append(f"مشتری احرازشده: کد ملی {state['national_code']}")
    if state.get("topic"):
        parts.append(f"آخرین موضوع گفتگو: {state['topic']}")
    recent = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-6:]
    if recent:
        parts.append("چند پیام اخیر:\n" + "\n".join(
            f"- {'کاربر' if isinstance(m, HumanMessage) else 'دستیار'}: {str(m.content)[:200]}"
            for m in recent if m.content
        ))
    return "\n".join(parts)


TOPIC_MAP = {
    "card_status": "وضعیت کارت",
    "new_card_request": "درخواست کارت جدید",
    "loan_eligibility": "وام",
    "policy_question": "قوانین بانک",
    "request_logging": "ثبت درخواست",
    "customer_info": "اطلاعات حساب",
}


def build_graph():
    llm = make_llm()
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # ---------------- nodes ------------------------------------------------
    def n_classify(state: AgentState) -> dict:
        msgs = state["messages"]
        last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        text = str(last_human.content) if last_human else ""
        known = state.get("national_code")
        result = classify_intent(llm, text, _context_summary(state))
        updates: dict = {}
        code = normalize_national_code(result.get("national_code")) if result.get("national_code") else None
        if code:
            updates["national_code"] = code
        elif known:
            result["national_code"] = known  # carry forward authenticated user
        intent = result.get("intent", "other")
        # deterministic rescue: classifier says clarify but context resolves referent
        if intent == "clarify" and state.get("topic") and (
            "کارت" in state["topic"] or "وضعیت" in state["topic"]
        ) and state.get("national_code"):
            intent = "card_status"
            result["needs_clarification"] = False
        updates["intent"] = intent
        updates["card_type"] = result.get("card_type") or state.get("card_type")
        updates["request_type"] = result.get("request_type")
        updates["needs_clarification"] = bool(result.get("needs_clarification"))
        updates["topic"] = TOPIC_MAP.get(intent, state.get("topic", ""))
        note = result.get("reasoning_note", "")
        updates["reasoning_log"] = (state.get("reasoning_log") or []) + [
            f"طبقه‌بندی قصد: {intent}" + (f" — {note}" if note else "")
        ]
        return updates

    def route_after_classify(state: AgentState) -> str:
        if state.get("needs_clarification") and not state.get("topic"):
            return "clarify"
        return "prepare_context"

    def n_clarify(state: AgentState) -> dict:
        msg = AIMessage(
            content=(
                "متوجه نشدم منظورتان دقیقاً چیست 🙂\n"
                "لطفاً بیشتر توضیح دهید: درباره وضعیت کارت است، درخواست وام، "
                "سؤال از قوانین، یا چیزی دیگر؟"
            )
        )
        return {"messages": [msg], "reasoning_log": (state.get("reasoning_log") or []) +
                ["ابهام در پیام کاربر بدون context → درخواست شفاف‌سازی"]}

    def n_prepare(state: AgentState) -> dict:
        from jalali_utils import today_jalali

        ctx = _context_summary(state)
        sys_prompt = (
            SYSTEM_CORE
            + f"\n\nامروز (تقویم جلالی): {today_jalali()}\n"
            + f"CONTEXT فعلی مکالمه:\n{ctx}\n"
            + f"قصد تشخیص‌داده‌شده این پیام: {state.get('intent')}"
        )
        new_msgs = list(state["messages"])
        # replace any previous injected system prompt to keep history clean
        new_msgs = [m for m in new_msgs if not isinstance(m, SystemMessage)]
        new_msgs.insert(0, SystemMessage(content=sys_prompt))
        return {"messages": new_msgs}

    def n_agent(state: AgentState) -> dict:
        resp = llm_with_tools.invoke(state["messages"])
        trace = (state.get("reasoning_log") or [])[:]
        if resp.tool_calls:
            calls = ", ".join(tc["name"] for tc in resp.tool_calls)
            trace.append(f"فراخوانی ابزار: {calls}")
        else:
            trace.append("تولید پاسخ نهایی برای کاربر")
        return {"messages": [resp], "reasoning_log": trace}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    def n_tools(state: AgentState) -> dict:
        out = tool_node.invoke({"messages": state["messages"]})
        return out

    # ---------------- graph --------------------------------------------------
    g = StateGraph(AgentState)
    g.add_node("classify_intent", n_classify)
    g.add_node("clarify", n_clarify)
    g.add_node("prepare_context", n_prepare)
    g.add_node("agent", n_agent)
    g.add_node("tools", n_tools)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route_after_classify,
                            {"clarify": "clarify", "prepare_context": "prepare_context"})
    g.add_edge("clarify", END)
    g.add_edge("prepare_context", "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "__end__": END})
    g.add_edge("tools", "agent")

    return g.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Public API used by CLI / Streamlit / tests
# ---------------------------------------------------------------------------
_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


DEFAULT_STATE = {
    "messages": [], "intent": "", "national_code": None, "card_type": None,
    "request_type": None, "needs_clarification": False, "topic": "",
    "reasoning_log": [],
}


def run_turn(user_text: str, thread_id: str = "default") -> dict:
    """Send one user message; returns final answer + reasoning trace."""
    graph = get_graph()
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=cfg)
    answer = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and m.content:
            answer = str(m.content)
            break
    return {
        "answer": answer,
        "reasoning": result.get("reasoning_log", []),
        "intent": result.get("intent", ""),
        "national_code": result.get("national_code"),
        "topic": result.get("topic", ""),
    }
