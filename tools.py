"""LangChain tool definitions for the Banking Agent.

Every tool returns a JSON string (compact, Persian keys) so the LLM can read
and re-narrate it faithfully. Tools implement:

* Requirement 1 : authentication & data access (get_customer, list_card_orders)
* Requirement 2 : RAG over policies          (search_policies)
* Requirement 3 : card order tracking        (list_card_orders / analyze timing)
* Requirement 4 : hybrid queries             (check_loan_eligibility combines
                  customer data + policy rules; temporal fields combine
                  card_orders + policies)
* Requirement 5 : request logging            (log_request with validations)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

import config
from data_access import (
    CardOrderRepo,
    CustomerNotFoundError,
    CustomerRepo,
    RequestLogger,
    evaluate_loan,
    membership_years,
    normalize_card_type,
    normalize_national_code,
    score_band,
    toman,
)
from jalali_utils import today_jalali
from rag import get_retriever

log = logging.getLogger("banking_agent.tools")

_customers: CustomerRepo | None = None
_orders: CardOrderRepo | None = None
_requests: RequestLogger | None = None


def repos():
    global _customers, _orders, _requests
    if _customers is None:
        _customers = CustomerRepo()
        _orders = CardOrderRepo()
        _requests = RequestLogger()
    return _customers, _orders, _requests


def _err(msg: str) -> str:
    return json.dumps({"خطا": True, "پیام": msg}, ensure_ascii=False)


def _ok(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tool 1 — customer lookup / authentication
# ---------------------------------------------------------------------------
@tool
def get_customer(national_code: str) -> str:
    """اطلاعات هویتی و مالی مشتری را از بانک اطلاعات مشتریان با کد ملی می‌خواند.
    Read the customer's identity/financial record (name, phone, balance in Toman,
    credit score 300-850, membership date in Jalali calendar, monthly transaction
    count) from customers.xlsx by their 10-digit Iranian national code.
    Returns an error if the code is invalid or the customer does not exist.
    Always call this first to identify/authenticate the user."""
    customers, _, _ = repos()
    code = normalize_national_code(national_code)
    if not code:
        return _err("کد ملی معتبر نیست؛ باید ۱۰ رقم باشد.")
    cust = customers.get(code)
    if cust is None:
        return _err(f"مشتری با کد ملی {code} در بانک اطلاعات وجود ندارد.")
    cust["سابقه عضویت (سال تقریبی)"] = membership_years(cust["تاریخ عضویت"])
    cust["طبقه امتیاز اعتباری"] = score_band(cust["امتیاز اعتباری"])
    cust["موجودی (متن)"] = toman(cust["موجودی"])
    return _ok({"یافت شد": True, "مشتری": cust})


# ---------------------------------------------------------------------------
# Tool 2 — card orders (tracking + temporal analysis)
# ---------------------------------------------------------------------------
@tool
def list_card_orders(
    national_code: str,
    only_active: bool = False,
    card_type: Optional[str] = None,
) -> str:
    """سفارش‌های کارت مشتری را از card_orders.xlsx برمی‌گرداند.
    List all card orders of a customer (credit/debit/gift cards) sorted newest
    first, each with: نوع کارت, تاریخ سفارش (Jalali), وضعیت (one of:
    در دست بررسی / تأیید شده / در حال چاپ / ارسال به پست / ارسال شده /
    تحویل داده شده), کد پیگیری, تاریخ تحویل تقریبی, آدرس ارسال — plus computed
    temporal fields: روز کاری سپری‌شده since the order, the standard issuance
    window from bank policy (بازه استاندارد صدور), موعد مجاز تحویل and whether
    the order is delayed (تأخیر دارد). Set only_active=true to keep orders that
    are not yet delivered; pass card_type ('اعتباری'/'نقدی'/'هدیه') to filter."""
    _, orders, _ = repos()
    code = normalize_national_code(national_code)
    if not code:
        return _err("کد ملی معتبر نیست.")
    result = orders.orders_for(code)
    if not result:
        return _ok({"سفارش‌ها": [], "تعداد": 0,
                    "یادداشت": "این مشتری هیچ سفارت کارتی ثبت نکرده است."})
    if only_active:
        result = [o for o in result if o["فعال"]]
    if card_type:
        fam = normalize_card_type(card_type)
        result = [o for o in result if o["خانواده کارت"] == fam]
    analyzed = [orders.analyze_order(o) for o in result]
    for a in analyzed:
        a.pop("_order_date", None)
        a.pop("_eta", None)
    return _ok({
        "سفارش‌ها": analyzed,
        "تعداد": len(analyzed),
        "چند سفارش فعال": sum(1 for a in analyzed if a.get("فعال")) > 1,
    })


# ---------------------------------------------------------------------------
# Tool 3 — RAG over bank policies
# ---------------------------------------------------------------------------
@tool
def search_policies(question: str) -> str:
    """جستجوی معنایی (RAG) در سند «قوانین و مقررات بانک دایا‌تدبیر».
    Semantic+keyword retrieval over the official bank policy document: loan
    conditions and interest rates, account types, card types/benefits/fees,
    daily withdrawal limits, issuance times (working days), authentication and
    security rules, support hours, complaint process. Pass the user's question
    (or the specific sub-topic) in Persian; returns the most relevant original
    policy passages verbatim."""
    retriever = get_retriever()
    passages = retriever.retrieve(question)
    if not passages:
        return _ok({"نتایج": [], "یادداشت": "مربوطی در قوانین پیدا نشد."})
    return _ok({"نتایج": passages, "تعداد": len(passages)})


# ---------------------------------------------------------------------------
# Tool 4 — hybrid loan eligibility (customer data + policy rules)
# ---------------------------------------------------------------------------
@tool
def check_loan_eligibility(national_code: str, loan_type: Optional[str] = None) -> str:
    """استحقاق دریافت وام را با ترکیب داده مشتری و قوانین بانک محاسبه می‌کند.
    Hybrid check combining the customer's credit score & balance (from
    customers.xlsx) with the bank's official loan rules (policies §3):
    قرض‌الحسنه (min score 700, ≥2 years history, ≥10M Toman avg balance),
    خرید خودرو (min score 650), مسکن (min score 700), کسب‌وکار (min score 680).
    Returns per-loan-type eligibility verdicts with reasons. Optionally filter
    one loan_type in Persian."""
    customers, _, _ = repos()
    code = normalize_national_code(national_code)
    if not code:
        return _err("کد ملی معتبر نیست.")
    cust = customers.get(code)
    if cust is None:
        return _err(f"مشتری با کد ملی {code} یافت نشد.")
    lt = loan_type.strip() if loan_type else None
    results = evaluate_loan(cust, lt)
    return _ok({
        "مشتری": {"نام": cust["نام"], "امتیاز اعتباری": cust["امتیاز اعتباری"],
                  "طبقه امتیاز": score_band(cust["امتیاز اعتباری"]),
                  "موجودی": toman(cust["موجودی"]),
                  "سابقه (سال)": membership_years(cust["تاریخ عضویت"])},
        "ارزیابی وام": results,
    })


# ---------------------------------------------------------------------------
# Tool 5 — request logging with validation (Req.5 + Constraint 5)
# ---------------------------------------------------------------------------
@tool
def log_request(
    national_code: str,
    request_type: str,
    description: str,
) -> str:
    """ثبت رسمی درخواست مشتری در requests.xlsx همراه با اعتبارسنجی.
    Log a customer request into requests.xlsx. request_type MUST be one of:
    وام / افتتاح حساب / سفارش کارت جدید / شکایت / استعلام / سایر.
    Validations performed automatically:
      1) the national code must exist in customers.xlsx;
      2) for 'سفارش کارت جدید': rejected if the customer already has an active
         (undelivered) card order of the same family OR an unprocessed new-card
         request logged today;
      3) duplicate warning if the same user logged the same request type today;
      4) creates requests.xlsx if missing. Returns the stored record."""
    customers, orders, requests = repos()
    code = normalize_national_code(national_code)
    if not code:
        return _err("کد ملی معتبر نیست.")
    rtype = request_type.strip()
    if rtype not in config.REQUEST_TYPES:
        return _err(f"نوع درخواست '{rtype}' نامعتبر است. مجاز: {' / '.join(config.REQUEST_TYPES)}")
    cust = customers.get(code)
    if cust is None:
        return _err(f"کد ملی {code} در customers.xlsx وجود ندارد؛ امکان ثبت درخواست نیست.")

    # validation 2 — new card order guard
    if rtype == "سفارش کارت جدید":
        fam = normalize_card_type(description)
        active = orders.active_orders(code)
        dup_active = [a for a in active if a["خانواده کارت"] == fam] if fam else active
        if dup_active:
            kinds = ", ".join(f"{a['نوع کارت']} ({a['وضعیت']})" for a in dup_active[:3])
            return _err(
                f"این کاربر هم‌اکنون سفارش کارت فعال دارد: {kinds}. "
                "طبق قانون، تا نتیجه‌نشدن سفارش قبلی، سفارش جدید همان نوع ثبت نمی‌شود."
            )
        if requests.has_active_new_card_request(code, fam if fam else None):
            return _err("درخواست «سفارش کارت جدید» برای این کاربر امروز ثبت شده و هنوز پردازش نشده است.")

    status = "ثبت شد"
    dup = requests.find_duplicate(code, rtype)
    if dup:
        status = "ثبت شد (هشدار: تکراری امروز)"

    rec = requests.log(code, cust["نام"], rtype, description, status=status)
    out = {"ثبت شد": True, "رکورد": rec, "تاریخ امروز": today_jalali()}
    if "تکراری" in status:
        out["هشدار"] = "همین کاربر همین نوع درخواست را امروز ثبت کرده بود."
    return _ok(out)


def get_tools():
    return [get_customer, list_card_orders, search_policies,
            check_loan_eligibility, log_request]


TOOL_MAP = {t.name: t for t in get_tools()}
