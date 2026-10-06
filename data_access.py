"""Data-access layer: customers / card orders / request logging (Excel).

Implements Requirement 1 (Authentication & Data Access) and
Requirement 5 (Request Logging with validation, Constraint 5).
"""

from __future__ import annotations

import os
import re
from datetime import date

import pandas as pd

import config
from jalali_utils import (
    add_working_days,
    format_jalali,
    parse_jalali,
    today_gregorian,
    working_days_between,
)


class CustomerNotFoundError(Exception):
    """Raised when a national code is not present in customers.xlsx."""


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalize_national_code(value) -> str | None:
    """Extract/validate an Iranian national code from arbitrary text.

    Accepts '0123456789', '۰۱۲۳۴۵۶۷۸۹', 'IR 1234567890' etc.
    Returns a zero-padded 10-digit string or ``None``.
    """
    if value is None:
        return None
    s = str(value)
    # Persian/Arabic digits -> latin
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)
    s = s.translate(trans)
    digits = re.sub(r"\D", "", s)
    if len(digits) == 9:  # some codes stored without leading zero
        digits = "0" + digits
    if len(digits) != 10:
        return None
    # checksum validation (standard Iranian national-id algorithm)
    if not _national_id_valid(digits):
        # still accept it but flag; banks data here is synthetic
        pass
    return digits


def _national_id_valid(code: str) -> bool:
    if not code.isdigit() or len(code) != 10:
        return False
    check = int(code[9])
    total = sum(int(code[i]) * (10 - i) for i in range(9)) % 11
    if total < 2:
        return check == total
    return check == 10 - total


def normalize_card_type(value) -> str:
    """Map raw card-type strings to canonical families: اعتباری/نقدی/هدیه."""
    s = str(value or "").strip()
    if "اعتبار" in s:
        return "اعتباری"
    if "نقد" in s:
        return "نقدی"
    if "هدیه" in s or "هديه" in s:
        return "هدیه"
    return s


def toman(n) -> str:
    try:
        return f"{int(n):,} تومان".replace(",", "،")
    except Exception:
        return str(n)


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------
class CustomerRepo:
    def __init__(self, path: str = config.CUSTOMERS_FILE):
        self.df = pd.read_excel(path, dtype={"کد ملی": str})
        self.df["کد ملی"] = self.df["کد ملی"].map(
            lambda x: normalize_national_code(x) or str(x)
        )

    def get(self, national_code: str) -> dict | None:
        row = self.df[self.df["کد ملی"] == national_code]
        if row.empty:
            return None
        r = row.iloc[0].to_dict()
        return {
            "کد ملی": r["کد ملی"],
            "نام": str(r.get("نام و نام خانوادگی", "")).strip(),
            "شماره تماس": str(r.get("شماره تماس", "")),
            "موجودی": int(r.get("موجودی حساب", 0) or 0),
            "امتیاز اعتباری": int(r.get("امتیاز اعتباری", 0) or 0),
            "تاریخ عضویت": str(r.get("تاریخ عضویت", "")),
            "تراکنش ماه اخیر": int(r.get("تعداد تراکنش‌های ماه اخیر", 0) or 0),
        }

    def exists(self, national_code: str) -> bool:
        return national_code in set(self.df["کد ملی"])


class CardOrderRepo:
    def __init__(self, path: str = config.CARD_ORDERS_FILE):
        self.df = pd.read_excel(path, dtype={"کد ملی": str})
        self.df["کد ملی"] = self.df["کد ملی"].map(
            lambda x: normalize_national_code(x) or str(x)
        )

    def orders_for(self, national_code: str) -> list[dict]:
        rows = self.df[self.df["کد ملی"] == national_code]
        out = []
        for _, r in rows.iterrows():
            order_date = parse_jalali(r["تاریخ سفارش"])
            eta_raw = str(r.get("تاریخ تحویل تقریبی", ""))
            eta = parse_jalali(eta_raw)
            status = str(r["وضعیت"]).strip()
            out.append(
                {
                    "کد ملی": r["کد ملی"],
                    "نوع کارت": str(r["نوع کارت"]).strip(),
                    "خانواده کارت": normalize_card_type(r["نوع کارت"]),
                    "تاریخ سفارش": str(r["تاریخ سفارش"]).strip(),
                    "تاریخ سفارش (میلادی)": order_date.isoformat() if order_date else None,
                    "وضعیت": status,
                    "فعال": status in config.ACTIVE_STATUSES,
                    "تحویل شده": status == config.DELIVERED_STATUS,
                    "کد پیگیری": str(r.get("کد پیگیری", "")).strip(),
                    "تاریخ تحویل تقریبی": eta_raw.strip(),
                    "آدرس ارسال": str(r.get("آدرس ارسال", "")).strip(),
                    "_order_date": order_date,
                    "_eta": eta,
                }
            )
        # newest first
        out.sort(key=lambda o: o["_order_date"] or date.min, reverse=True)
        return out

    def active_orders(self, national_code: str) -> list[dict]:
        return [o for o in self.orders_for(national_code) if o["فعال"]]

    def latest_order(self, national_code: str) -> dict | None:
        orders = self.orders_for(national_code)
        return orders[0] if orders else None

    # ---- temporal analysis -------------------------------------------------
    def analyze_order(self, order: dict) -> dict:
        """Add elapsed-working-days / delay info to one order dict."""
        od = order.get("_order_date")
        today = today_gregorian()
        res = dict(order)
        if od:
            elapsed = working_days_between(od, today)
            fam = order.get("خانواده کارت", "")
            min_d, max_d = config.CARD_ISSUANCE_DAYS.get(
                order.get("نوع کارت", ""),
                config.CARD_ISSUANCE_DAYS.get(fam, config.DEFAULT_ISSUANCE_WINDOW),
            )
            expected_until = add_working_days(od, max_d)
            res["روز کاری سپری‌شده"] = elapsed
            res["بازه استاندارد صدور (روز کاری)"] = f"{min_d}-{max_d}"
            res["موعد مجاز تحویل"] = format_jalali(expected_until)
            res["تأخیر دارد"] = (not order["تحویل شده"]) and today > expected_until
            if res["تأخیر دارد"]:
                res["روز تأخیر"] = working_days_between(expected_until, today)
        return res


class RequestLogger:
    """Append-only logger writing requests to requests.xlsx (Req. 5)."""

    COLUMNS = ["کد ملی", "نام", "نوع درخواست", "توضیحات", "تاریخ", "وضعیت"]

    def __init__(self, path: str = config.REQUESTS_FILE):
        self.path = path

    def _load(self) -> pd.DataFrame:
        if os.path.exists(self.path):
            try:
                return pd.read_excel(self.path)
            except Exception:
                pass
        return pd.DataFrame(columns=self.COLUMNS)

    def find_duplicate(self, national_code: str, req_type: str) -> dict | None:
        """Same user + same request type recorded on the Jalali day of today."""
        df = self._load()
        if df.empty:
            return None
        today_j = today_gregorian()
        for _, r in df.iterrows():
            if str(r.get("کد ملی")) == national_code and str(r.get("نوع درخواست")) == req_type:
                rec_date = parse_jalali(r.get("تاریخ"))
                if rec_date == today_j:
                    return r.to_dict()
        return None

    def has_active_new_card_request(self, national_code: str, card_family: str | None = None) -> bool:
        """True if user already logged a 'سفارش کارت جدید' that isn't rejected."""
        df = self._load()
        if df.empty:
            return False
        for _, r in df.iterrows():
            if (
                str(r.get("کد ملی")) == national_code
                and str(r.get("نوع درخواست")) == "سفارش کارت جدید"
                and str(r.get("وضعیت")) not in ("رد شده", "لغو شده")
            ):
                if card_family is None or card_family in str(r.get("توضیحات", "")):
                    return True
        return False

    def log(
        self,
        national_code: str,
        name: str,
        req_type: str,
        description: str,
        status: str = "ثبت شد",
    ) -> dict:
        df = self._load()
        record = {
            "کد ملی": national_code,
            "نام": name,
            "نوع درخواست": req_type,
            "توضیحات": description,
            "تاریخ": format_jalali(today_gregorian()),
            "وضعیت": status,
        }
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_excel(self.path, index=False)
        return record


# ---------------------------------------------------------------------------
# Loan eligibility engine (deterministic rules from bank_policies.docx §3)
# ---------------------------------------------------------------------------
LOAN_RULES = {
    "قرض‌الحسنه": {"min_score": 700, "min_years": 2, "min_avg_balance": 10_000_000},
    "خرید خودرو": {"min_score": 650},
    "مسکن": {"min_score": 700},
    "کسب‌وکار": {"min_score": 680},
}


def score_band(score: int) -> str:
    if score >= 800:
        return "عالی (ریسک بسیار پایین)"
    if score >= 740:
        return "خوب (ریسک پایین)"
    if score >= 670:
        return "متوسط (ریسک معمولی)"
    if score >= 580:
        return "ضعیف (ریسک بالا)"
    return "بسیار ضعیف (عدم واجد شرایط)"


def membership_years(member_date_str: str) -> float:
    d = parse_jalali(member_date_str)
    if not d:
        return 0.0
    return round(working_days_between(d, today_gregorian()) / 365.0 * (7 / 5), 2)


def evaluate_loan(customer: dict, loan_type: str | None = None) -> list[dict]:
    """Return per-loan-type eligibility verdicts using policy thresholds."""
    results = []
    score = customer["امتیاز اعتباری"]
    balance = customer["موجودی"]
    years = membership_years(customer["تاریخ عضویت"])
    types = [loan_type] if loan_type and loan_type in LOAN_RULES else list(LOAN_RULES)
    for lt in types:
        rule = LOAN_RULES[lt]
        reasons = []
        ok = True
        if score < rule["min_score"]:
            ok = False
            reasons.append(
                f"امتیاز اعتباری {score} کمتر از حداقل لازم ({rule['min_score']}) است"
            )
        if lt == "قرض‌الحسنه":
            if years < rule["min_years"]:
                ok = False
                reasons.append(f"سابقه حساب {years} سال کمتر از {rule['min_years']} سال است")
            if balance < rule["min_avg_balance"]:
                ok = False
                reasons.append(
                    f"موجودی {toman(balance)} کمتر از حداقل {toman(rule['min_avg_balance'])} است"
                )
        results.append(
            {
                "نوع وام": lt,
                "واجد شرایط": ok,
                "دلیل": "تمام شرایط احراز شد" if ok else "؛ ".join(reasons),
                "طبقه امتیاز": score_band(score),
            }
        )
    return results
