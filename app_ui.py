"""Streamlit UI for the Persian Banking Agent (bonus requirement).

Run:  streamlit run app_ui.py
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(page_title="دستیار بانکی دایا‌تدبیر", layout="wide")
st.markdown(
    "<style>html{direction:rtl;text-align:right}</style>", unsafe_allow_html=True
)

st.title("🏦 دستیار هوشمند بانک دایا‌تدبیر")
st.caption("LangGraph + GPT-4o (OpenAI API) + Persian embeddings RAG")

# ---------------------------------------------------------------- state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:8]
if "chat" not in st.session_state:
    st.session_state.chat = []
if "trace" not in st.session_state:
    st.session_state.trace = {}
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

with st.sidebar:
    st.header("⚙️ تنظیمات")
    st.write(f"شناسه گفتگو: `{st.session_state.thread_id}`")
    if st.button("🔄 شروع گفتگوی جدید"):
        st.session_state.chat = []
        st.session_state.trace = {}
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.rerun()

    st.divider()
    st.subheader("📝 درخواست‌های ثبت‌شده")
    if os.path.exists("requests.xlsx"):
        df = pd.read_excel("requests.xlsx")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("هنوز درخواستی ثبت نشده است.")

    st.divider()
    with st.expander("💡 مثال سؤالات"):
        for ex in [
            "سلام، کد ملی من 0221536568 هست. کارتم چی شد؟",
            "چرا کارتم بعد از ۱۰ روز هنوز نرسیده؟",
            "وضعیتش چیه؟",
            "آیا می‌تونم وام بگیرم؟",
            "یه کارت اعتباری جدید میخوام، چقدر طول میکشه؟",
            "کارتم گم شده، میخوام شکایت کنم",
            "محدودیت برداشت روزانه کارت نقدی طلایی چقدره؟",
        ]:
            st.markdown(f"- {ex}")

# ---------------------------------------------------------------- init agent
with st.spinner("در حال آماده‌سازی مدل و نمایه‌سازی قوانین (بار اول ممکن است چند دقیقه طول بکشد)..."):
    try:
        from agent import run_turn  # builds graph lazily inside

        # build the RAG index once up-front (heavy: downloads Persian model)
        import rag

        rag.get_retriever()
        st.session_state.agent_ready = True
    except Exception as exc:
        st.error(f"راه‌اندازی agent ناموفق بود: {exc}\n\n"
                 "لطفاً OPENAI_API_KEY را در فایل .env تنظیم کنید.")

# ---------------------------------------------------------------- chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🧠 مراحل تفکر عامل (reasoning trace)"):
                for i, t in enumerate(msg["trace"], 1):
                    st.markdown(f"{i}. {t}")

prompt = st.chat_input("پیام خود را به فارسی بنویسید…")
if prompt and st.session_state.agent_ready:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            res = run_turn(prompt, thread_id=st.session_state.thread_id)
            answer = res["answer"] or "(پاسخی تولید نشد)"
            trace = res.get("reasoning", [])
            with st.expander("🧠 مراحل تفکر عامل (reasoning trace)"):
                for i, t in enumerate(trace, 1):
                    st.markdown(f"{i}. {t}")
            st.markdown(answer)
            st.session_state.chat.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except Exception as exc:
            err = f"❌ خطا در پردازش: {exc}"
            st.error(err)
            st.session_state.chat.append({"role": "assistant", "content": err, "trace": []})
