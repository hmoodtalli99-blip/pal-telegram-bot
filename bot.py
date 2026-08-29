"""
PAL v12.0 - Structural Edition
================================
بوت تحليل وتنبيهات (Paper/Signal Bot عبر Telegram) معاد هيكلته بالكامل.

ملاحظة منهجية مهمة (اقرأها قبل التشغيل):
- هذا البوت أداة تحليل وتنبيه، وليس نظام تنفيذ آلي فعلي على وسيط حقيقي.
- كل ما يتعلق بـ ICT/SMC/Elliott Wave هو أطر وصفية شائعة في مجتمع التداول
  وليست خوارزميات موثقة أكاديمياً. تم بناؤها هنا كدوال حتمية قابلة للقياس
  (structure/OB/FVG قابلة للتحقق 100%)، لكن قوتها التنبؤية غير مضمونة.
  يجب اختبارها إحصائياً (backtesting) على بيانات كل أصل قبل الاعتماد عليها.
- Elliott Wave وNamespace التعلم التكيفي هنا معلوماتيان بشكل أساسي (context)
  وليسا مصدر قرار وحيد - محاولة تصميمهما هكذا عمداً لتفادي الإفراط في الثقة.

النشر على Railway:
1. ارفع هذا المجلد (bot.py + requirements.txt + Procfile) على GitHub.
2. اربط المستودع بمشروع جديد على Railway.
3. أضف متغيرات البيئة التالية في Railway > Variables:
   TELEGRAM_TOKEN, FINNHUB_KEY, DATABASE_URL, CHAT_ID
4. أضف قاعدة بيانات Postgres من Railway (يعطيك DATABASE_URL تلقائياً).
5. Railway سيشغّل Procfile تلقائياً (worker: python bot.py).
"""

import os
import io
import sys
import time
import math
import logging
import threading
import datetime
from dataclasses import dataclass, field
from typing import Optional

import pytz
import requests
import psycopg2
import numpy as np
import pandas as pd
import telebot
from ta.volatility import AverageTrueRange

# matplotlib بدون واجهة رسومية (سيرفر بلا شاشة على Railway)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf

# المنطق الحتمي (بنية/SMC/زخم) بات في ملف مستقل بلا أي أثر جانبي، حتى يستخدمه
# backtester.py وtest_signals.py بدون تشغيل بوت أو قاعدة بيانات فعليَين.
from signals import (
    find_swings, market_structure_state, get_dynamic_sr_zones, regime_filter,
    detect_fvg, detect_order_blocks, detect_liquidity_sweep,
    elliott_wave_context, structure_signal, smc_signal, momentum_signal,
    evaluate_signal,
)

# ========================================================================
# 0. التسجيل (Logging) - بديل print العشوائي
# ========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("PAL")

# ========================================================================
# 1. الإعدادات + فحص صارم عند الإقلاع (كان غائباً تماماً سابقاً)
# ========================================================================
REQUIRED_ENV = ["TELEGRAM_TOKEN", "FINNHUB_KEY", "DATABASE_URL", "CHAT_ID"]
missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if missing:
    log.critical(f"متغيرات بيئة ناقصة: {missing} - أوقف التشغيل وتحقق من إعدادات Railway.")
    sys.exit(1)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
FINNHUB_KEY = os.environ["FINNHUB_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
CHAT_ID = os.environ["CHAT_ID"]

# تفعيل/تعطيل مكونات اختيارية بدون تعديل الكود
ENABLE_CHARTS = os.environ.get("ENABLE_CHARTS", "true").lower() == "true"
ENABLE_ECON_CALENDAR = os.environ.get("ENABLE_ECON_CALENDAR", "false").lower() == "true"
ENABLE_META_MODEL = os.environ.get("ENABLE_META_MODEL", "false").lower() == "true"

bot = telebot.TeleBot(TELEGRAM_TOKEN)
GZA = pytz.timezone("Asia/Gaza")

DEMO_BALANCE_START = 50.0
BASE_RISK_PERCENT = 0.01          # مخاطرة أساسية لكل صفقة (1% من الرصيد)
MAX_LOT = 0.5                     # سقف صارم للوت (كان غائباً تماماً)
MAX_TOTAL_EXPOSURE_PCT = 0.05     # سقف مخاطرة كلية مفتوحة في آن واحد (5% من الرصيد)
DAILY_LOSS_CIRCUIT_BREAKER = 0.06 # إيقاف تام (وليس تخفيف) عند خسارة 6% باليوم
SPREAD_BUFFER = 0.0003
MAX_SLIPPAGE = 0.0015             # أُعيد تشديده (كان تم تخفيفه سابقاً)
MIN_RR_RATIO = 1.8                # أُعيد رفعه (كان خُفّض إلى 1.5 لزيادة القبول)
ADAPTIVE_MIN_SAMPLE = 30          # كان 3 فقط - رقم غير دال إحصائياً

TEAM_RULES = {
    "sniper": {"lock_profit": 0.15, "target1": 0.3, "risk_mult": 0.5},
    "scalp":  {"lock_profit": 0.5,  "target1": 1.0, "risk_mult": 1.0},
    "daily":  {"lock_profit": 2.0,  "target1": 4.0, "risk_mult": 1.2},
    "swing":  {"lock_profit": 5.0,  "target1": 10.0, "risk_mult": 1.5},
}

ASSETS_LIST = [
    {"key": "XAUUSD", "name": "الذهب", "group": "METAL"},
    {"key": "EURUSD", "name": "يورو دولار", "group": "FOREX_USD"},
    {"key": "GBPUSD", "name": "باوند دولار", "group": "FOREX_USD"},
]

# قفل مزامنة صريح - كان غائباً تماماً رغم 4+ خيوط تتشارك OPEN_TRADES
STATE_LOCK = threading.RLock()
OPEN_TRADES = {}
BOT_ACTIVE = True

SIGNAL_WEIGHTS = {
    "structure": 1.0,
    "smc": 1.0,
    "momentum": 1.0,
}

# ========================================================================
# 2. قاعدة البيانات
# ========================================================================
def db_connect():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def db_setup():
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY, inst TEXT, team TEXT, dir TEXT,
            result TEXT, profit FLOAT, time TIMESTAMP, report TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS brain (
            id INT PRIMARY KEY, value REAL, wins INT DEFAULT 0, losses INT DEFAULT 0)""")
        cur.execute(
            "INSERT INTO brain (id, value) VALUES (1, %s) ON CONFLICT (id) DO NOTHING",
            (DEMO_BALANCE_START,),
        )
        conn.commit()
        cur.close()
        conn.close()
        log.info("DB setup OK")
    except Exception as e:
        log.exception(f"DB Setup Error: {e}")


def get_balance():
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT value FROM brain WHERE id=1")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else DEMO_BALANCE_START
    except Exception as e:
        log.warning(f"get_balance fallback: {e}")
        return DEMO_BALANCE_START


def update_balance(change):
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("UPDATE brain SET value = value + %s WHERE id=1", (change,))
        if change > 0:
            cur.execute("UPDATE brain SET wins = wins + 1 WHERE id=1")
        else:
            cur.execute("UPDATE brain SET losses = losses + 1 WHERE id=1")
        conn.commit()
        cur.close()
        conn.close()
        evaluate_adaptive_learning()
    except Exception as e:
        log.exception(f"Update Balance Error: {e}")


def save_trade(inst, team, direction, result, profit, report):
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO trades (inst, team, dir, result, profit, time, report) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (inst, team, direction, result, profit, datetime.datetime.now(GZA), report),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        log.exception(f"Save Trade Error: {e}")


def get_today_loss():
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT SUM(profit) FROM trades WHERE time > CURRENT_DATE AND profit < 0")
        res = cur.fetchone()[0]
        cur.close()
        conn.close()
        return abs(res) if res else 0.0
    except Exception:
        return 0.0


# ========================================================================
# 3. جلب البيانات (مع إعادة محاولة - كان يفشل بصمت)
# ========================================================================
def fetch_biquote(symbol, timeframe, limit=200, retries=2):
    for attempt in range(retries + 1):
        try:
            if symbol == "XAUUSD":
                url = (f"https://api.finnhub.io/api/v1/crypto/candle?symbol=BINANCE:XAUUSD"
                       f"&resolution={timeframe}&count={limit}&token={FINNHUB_KEY}")
            else:
                url = (f"https://api.finnhub.io/api/v1/forex/candle?symbol=OANDA:{symbol}"
                       f"&resolution={timeframe}&count={limit}&token={FINNHUB_KEY}")
            res = requests.get(url, timeout=10)
            data = res.json()
            if data.get("s") != "ok" or not data.get("c"):
                return pd.DataFrame()
            df = pd.DataFrame({
                "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"],
                "t": pd.to_datetime(data["t"], unit="s"),
            })
            return df
        except Exception as e:
            log.warning(f"Fetch attempt {attempt+1} failed for {symbol}: {e}")
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def send_telegram(msg, photo_buf=None):
    try:
        if photo_buf is not None:
            bot.send_photo(CHAT_ID, photo_buf, caption=msg, parse_mode="HTML")
        else:
            bot.send_message(CHAT_ID, msg, parse_mode="HTML")
    except Exception as e:
        log.exception(f"Telegram Error: {e}")


# ========================================================================
# 4. جلسات التداول (سياق زمني - لا يُضاف كصوت، بل يضبط الحجم فقط)
# ========================================================================
def get_session_context(now_utc=None):
    now_utc = now_utc or datetime.datetime.utcnow()
    h = now_utc.hour
    if 7 <= h < 12:
        return "LONDON", 1.0
    if 12 <= h < 16:
        return "LONDON_NY_OVERLAP", 1.2
    if 16 <= h < 20:
        return "NEW_YORK", 1.0
    if 0 <= h < 7:
        return "ASIA", 0.5
    return "OFF_HOURS", 0.4


# ========================================================================
# ملاحظة: دوال بنية السوق / SMC / الزخم (find_swings, market_structure_state,
# get_dynamic_sr_zones, regime_filter, detect_fvg, detect_order_blocks,
# detect_liquidity_sweep, elliott_wave_context, structure_signal, smc_signal,
# momentum_signal) انتقلت بالكامل إلى signals.py - مستوردة أعلى الملف.
# ========================================================================


# ========================================================================
# 9. التعلّم التكيفي (رُفع الحد الأدنى للعيّنة من 3 إلى 30 - إحصائياً غير دال قبل ذلك)
# ========================================================================
def evaluate_adaptive_learning():
    global SIGNAL_WEIGHTS
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT report, result FROM trades ORDER BY id DESC LIMIT 150")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if not rows:
            return

        stats = {"structure": [0, 0], "smc": [0, 0], "momentum": [0, 0]}
        for report, res in rows:
            is_win = res == "WIN"
            for key in stats:
                if key in (report or "").lower():
                    stats[key][0] += 1
                    if is_win:
                        stats[key][1] += 1

        for key, (total, wins) in stats.items():
            if total >= ADAPTIVE_MIN_SAMPLE:
                win_rate = wins / total
                # نطاق ضيق (0.85-1.15) - تعديل طفيف فقط، ليس انقلاباً كاملاً
                if win_rate < 0.35:
                    SIGNAL_WEIGHTS[key] = 0.85
                elif win_rate > 0.55:
                    SIGNAL_WEIGHTS[key] = 1.15
                else:
                    SIGNAL_WEIGHTS[key] = 1.0
    except Exception as e:
        log.exception(f"Adaptive Learning Error: {e}")


# ========================================================================
# 10. التقويم الاقتصادي - Stub معطّل افتراضياً (كان غائباً تماماً)
# ========================================================================
def is_high_impact_news_window():
    """
    TODO: اربطها بمصدر تقويم اقتصادي حقيقي (مثال: Finnhub Economic Calendar
    أو ForexFactory API) عبر متغير بيئة ECON_CALENDAR_KEY.
    حالياً: مُعطّلة افتراضياً (ENABLE_ECON_CALENDAR=false) لتفادي إعطاء وهم
    حماية غير موجودة فعلياً.
    """
    if not ENABLE_ECON_CALENDAR:
        return False
    # -- نقطة ربط مستقبلية --
    return False


# ========================================================================
# 11. الرسم البصري (خارطة الطريق المرئية لكل حركة)
# ========================================================================
def render_chart(df, symbol, structure_state, smc_data, entry=None, sl=None, tp=None):
    if not ENABLE_CHARTS or len(df) < 20:
        return None
    try:
        plot_df = df.tail(80).copy()
        plot_df.index = pd.to_datetime(plot_df["t"])
        plot_df = plot_df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close"})

        addplots = []
        hlines = dict(hlines=[], colors=[], linestyle="--", linewidths=0.8)

        for zone in get_dynamic_sr_zones(df)[:4]:
            hlines["hlines"].append(zone["mid"])
            hlines["colors"].append("gray")

        if entry:
            hlines["hlines"].append(entry)
            hlines["colors"].append("blue")
        if sl:
            hlines["hlines"].append(sl)
            hlines["colors"].append("red")
        if tp:
            hlines["hlines"].append(tp)
            hlines["colors"].append("green")

        buf = io.BytesIO()
        mpf.plot(
            plot_df, type="candle", style="charles",
            title=f"{symbol} - {structure_state['trend']}",
            hlines=hlines if hlines["hlines"] else None,
            savefig=dict(fname=buf, dpi=110, bbox_inches="tight"),
        )
        buf.seek(0)
        return buf
    except Exception as e:
        log.exception(f"Chart render error: {e}")
        return None


# ========================================================================
# 12. الارتباط الفعلي بين الأصول (rolling correlation - بدل فرضية ثابتة)
# ========================================================================
_price_cache = {}

def update_price_cache(symbol, df):
    _price_cache[symbol] = df["c"].tail(50).reset_index(drop=True)


def get_correlation(symbol_a, symbol_b):
    if symbol_a not in _price_cache or symbol_b not in _price_cache:
        return 0.0
    a, b = _price_cache[symbol_a], _price_cache[symbol_b]
    n = min(len(a), len(b))
    if n < 10:
        return 0.0
    return float(a.tail(n).corr(b.tail(n)))


def check_correlation_and_exposure(new_asset):
    """
    استبدال الفحص السابق (يستثني METAL بشكل تعسفي) بفحص ارتباط فعلي محسوب
    + سقف تعرض كلي عبر كل الأصول، وليس فقط ضمن نفس "المجموعة".
    """
    with STATE_LOCK:
        open_list = list(OPEN_TRADES.values())

    # سقف التعرض الكلي (المخاطرة المفتوحة كنسبة من الرصيد)
    total_open_risk = sum(t.get("risk_amount", 0) for t in open_list)
    if total_open_risk >= get_balance() * MAX_TOTAL_EXPOSURE_PCT:
        return False, "سقف التعرض الكلي للمحفظة"

    # ارتباط فعلي محسوب (وليس فرضية ثابتة تستثني الذهب)
    for t in open_list:
        corr = get_correlation(new_asset["key"], t["inst"])
        if abs(corr) >= 0.75:
            return False, f"ارتباط مرتفع محسوب ({corr:.2f}) مع صفقة {t['inst']} مفتوحة"

    return True, "OK"


# ========================================================================
# 13. محرّك القرار المركزي (King) - يدمج البنية + SMC + الزخم + فلتر النظام
# ========================================================================
def King_Brain(inst, team="daily"):
    if not BOT_ACTIVE:
        return
    symbol = inst["key"]

    if is_high_impact_news_window():
        log.info(f"{symbol}: تخطي بسبب نافذة أخبار عالية التأثير")
        return

    ok, reason = check_correlation_and_exposure(inst)
    if not ok:
        log.info(f"{symbol}: تخطي - {reason}")
        return

    tf_map = {
        "sniper": ("5", "1", 0.3, 3),
        "scalp": ("60", "5", 1.0, 3),
        "daily": ("240", "15", 2.0, 3),
        "swing": ("D", "60", 3.0, 4),
    }
    tf_trend, tf_entry, tp_mul, base_threshold = tf_map[team]

    df_trend = fetch_biquote(symbol, tf_trend, 200)
    df_entry = fetch_biquote(symbol, tf_entry, 200)
    if len(df_trend) < 50 or len(df_entry) < 50:
        return

    update_price_cache(symbol, df_entry)

    current_open = df_entry["o"].iloc[-1]
    current_close = df_entry["c"].iloc[-1]
    slippage = abs(current_close - current_open) / current_open
    if slippage > MAX_SLIPPAGE:
        return

    # القرار عبر evaluate_signal الموحّدة في signals.py - نفس الدالة بالحرف
    # يستخدمها backtester.py، لضمان ألا ينحرف منطق التشغيل الحي عن منطق
    # الاختبار التاريخي (وهي مشكلة شائعة تُبطل نتائج أي backtest سابق عليها).
    decision = evaluate_signal(df_trend, df_entry, base_threshold, weights=SIGNAL_WEIGHTS)
    ew_context = elliott_wave_context(df_trend)  # معلوماتي فقط، لا يدخل في القرار

    report = (
        f"<b>[{team.upper()}] {symbol}</b>\n"
        f"نظام السوق: {decision['regime']}\n"
        f"structure: {decision['struct_label']}\n"
        f"smc: {decision['smc_label']}\n"
        f"momentum: {decision['mom_label']}\n"
        f"موجات (معلوماتي): {ew_context['count']}"
    )

    if decision["direction"] is None:
        return

    atr = AverageTrueRange(high=df_entry["h"], low=df_entry["l"], close=df_entry["c"], window=14)
    atr_val = atr.average_true_range().iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return

    if decision["direction"] == "BUY":
        entry = current_close
        sl = entry - atr_val * 1.2
        tp = entry + atr_val * 1.2 * tp_mul
    else:
        entry = current_close
        sl = entry + atr_val * 1.2
        tp = entry - atr_val * 1.2 * tp_mul

    feedback_loop(symbol, decision["direction"], entry, sl, tp, report, team,
                  df_entry, decision["struct_state"], decision["smc_data"])


def feedback_loop(inst, direction, entry, sl, tp, report, team, df_entry, struct_state, smc_data):
    if abs(entry - sl) == 0:
        return
    rr = abs(tp - entry) / abs(entry - sl)
    if rr < MIN_RR_RATIO:
        return

    # دورة اليوم: قاطع تام عند تجاوز الحد اليومي (وليس تخفيف 0.7x كما سابقاً)
    if get_today_loss() >= get_balance() * DAILY_LOSS_CIRCUIT_BREAKER:
        log.info("تم تفعيل قاطع الخسارة اليومي - لا صفقات جديدة اليوم")
        return

    session_name, session_mult = get_session_context()
    team_risk_mult = TEAM_RULES[team]["risk_mult"]  # أصبحت مُستخدَمة فعلياً (كانت ميتة سابقاً)

    risk_pct = BASE_RISK_PERCENT * team_risk_mult * session_mult
    balance = get_balance()
    risk_amount = balance * risk_pct
    lot = round(risk_amount / (abs(entry - sl) * 100), 2)
    lot = min(lot, MAX_LOT)  # سقف صارم - كان غائباً تماماً
    if lot <= 0:
        return

    trade_id = f"{inst}_{team}_{int(time.time())}"
    trade = {
        "inst": inst, "dir": direction, "entry": entry, "sl": sl, "tp": tp,
        "lot": lot, "team": team, "report": report, "stage": 0,
        "risk_amount": risk_amount, "session": session_name,
    }

    with STATE_LOCK:
        OPEN_TRADES[trade_id] = trade

    chart_buf = render_chart(df_entry, inst, struct_state, smc_data, entry, sl, tp)
    caption = (
        f"👑 [{team}] {direction} {inst}\n"
        f"الجلسة: {session_name} | Lot: {lot}\n"
        f"Entry: {entry:.5f}\nSL: {sl:.5f}\nTP: {tp:.5f}\nR:R = {rr:.2f}\n\n{report}"
    )
    send_telegram(caption, photo_buf=chart_buf)

    threading.Thread(target=monitor_trade, args=(trade_id,), daemon=True).start()


def monitor_trade(trade_id):
    with STATE_LOCK:
        trade = OPEN_TRADES.get(trade_id)
    if not trade:
        return
    team = trade["team"]
    rules = TEAM_RULES[team]

    while True:
        with STATE_LOCK:
            if trade_id not in OPEN_TRADES:
                return
            trade = OPEN_TRADES[trade_id]

        time.sleep(30)
        df = fetch_biquote(trade["inst"], "1" if team == "sniper" else "5", 5)
        if not len(df):
            continue

        price = df["c"].iloc[-1]
        profit_now = price - trade["entry"] if trade["dir"] == "BUY" else trade["entry"] - price

        with STATE_LOCK:
            if trade_id not in OPEN_TRADES:
                return
            trade = OPEN_TRADES[trade_id]

            if trade["stage"] == 0 and profit_now >= rules["lock_profit"]:
                new_sl = ((trade["entry"] + rules["lock_profit"] + SPREAD_BUFFER) if trade["dir"] == "BUY"
                          else (trade["entry"] - rules["lock_profit"] - SPREAD_BUFFER))
                trade["sl"] = new_sl
                trade["stage"] = 1
                send_telegram(f"🛡️ [{team}] تأمين SL -> {new_sl:.5f}")

            if trade["stage"] == 1 and profit_now >= rules["target1"]:
                profit = rules["target1"] * trade["lot"] * 100 * 0.5
                update_balance(profit)
                trade["sl"] = (trade["entry"] + rules["lock_profit"] if trade["dir"] == "BUY"
                               else trade["entry"] - rules["lock_profit"])
                trade["stage"] = 2
                send_telegram(f"🎯 [{team}] هدف أول +${profit:.2f}")

            closed = False
            if trade["dir"] == "BUY":
                if price >= trade["tp"]:
                    profit = (trade["tp"] - trade["entry"]) * trade["lot"] * 100
                    update_balance(profit)
                    save_trade(trade["inst"], team, trade["dir"], "WIN", profit, trade["report"])
                    closed = True
                elif price <= trade["sl"]:
                    profit = (trade["sl"] - trade["entry"]) * trade["lot"] * 100
                    update_balance(profit)
                    save_trade(trade["inst"], team, trade["dir"], "LOSS", profit, trade["report"])
                    closed = True
            else:
                if price <= trade["tp"]:
                    profit = (trade["entry"] - trade["tp"]) * trade["lot"] * 100
                    update_balance(profit)
                    save_trade(trade["inst"], team, trade["dir"], "WIN", profit, trade["report"])
                    closed = True
                elif price >= trade["sl"]:
                    profit = (trade["entry"] - trade["sl"]) * trade["lot"] * 100
                    update_balance(profit)
                    save_trade(trade["inst"], team, trade["dir"], "LOSS", profit, trade["report"])
                    closed = True

            if closed:
                del OPEN_TRADES[trade_id]
                return


# ========================================================================
# 14. دوائر العمل الخلفية + مراقبة صحة النظام (Heartbeat)
# ========================================================================
# آخر وقت نجاح فعلي لكل حلقة - يكتشف الأعطال الصامتة (حلقة عالقة أو ميتة
# دون استثناء واضح تراه في السجلات). كان غائباً تماماً سابقاً.
LOOP_HEARTBEATS = {"sniper": None, "scalp": None, "daily": None, "swing": None}
HEARTBEAT_LOCK = threading.Lock()
HEARTBEAT_STALE_MINUTES = 20  # إن لم تنبض حلقة خلال هذه المدة، اعتبرها متوقفة


def run_loop(team, per_asset_sleep, cycle_sleep):
    while True:
        if BOT_ACTIVE:
            for asset in ASSETS_LIST:
                try:
                    King_Brain(asset, team)
                except Exception as e:
                    log.exception(f"King_Brain error [{team}/{asset['key']}]: {e}")
                time.sleep(per_asset_sleep)
            with HEARTBEAT_LOCK:
                LOOP_HEARTBEATS[team] = datetime.datetime.now(GZA)
        time.sleep(cycle_sleep)


def heartbeat_monitor(check_every_minutes=30):
    """
    يفحص دورياً أن الحلقات الأربعة ما زالت تعمل، ويرسل تنبيهاً صريحاً على
    Telegram إن توقفت إحداها بصمت - بدل اكتشاف العطل بعد أيام صدفة.
    """
    time.sleep(120)  # مهلة إقلاع أولية قبل أول فحص
    while True:
        time.sleep(check_every_minutes * 60)
        now = datetime.datetime.now(GZA)
        with HEARTBEAT_LOCK:
            snapshot = dict(LOOP_HEARTBEATS)
        stalled = []
        for team, last_seen in snapshot.items():
            if last_seen is None:
                continue  # لم يكتمل أول دورة بعد، طبيعي عند الإقلاع
            minutes_since = (now - last_seen).total_seconds() / 60
            if minutes_since > HEARTBEAT_STALE_MINUTES:
                stalled.append(f"{team} (منذ {minutes_since:.0f} دقيقة)")
        if stalled:
            send_telegram(f"⚠️ <b>تحذير صحة النظام:</b> حلقات متوقفة عن النبض:\n" + "\n".join(stalled))
        else:
            log.info(f"Heartbeat OK: {snapshot}")


# ========================================================================
# 15. واجهة تليجرام
# ========================================================================
@bot.message_handler(commands=["start"])
def start(m):
    send_telegram(
        "👑 <b>PAL v12.0 (Structural Edition)</b>\n"
        "تحليل قائم على بنية السوق + SMC/ICT + الزخم، مع فلتر نظام وضوابط مخاطر صارمة.\n"
        "هذا بوت تحليل/تنبيه، وليس تنفيذاً آلياً فعلياً على وسيط حقيقي."
    )

@bot.message_handler(commands=["balance"])
def balance(m):
    send_telegram(f"💰 الرصيد الحالي: ${get_balance():.2f}")

@bot.message_handler(commands=["active"])
def active_trades(m):
    with STATE_LOCK:
        items = list(OPEN_TRADES.items())[-5:]
    if not items:
        send_telegram("📂 لا توجد صفقات مفتوحة حالياً.")
        return
    msg = "📊 <b>الصفقات المفتوحة:</b>\n\n"
    for tid, t in items:
        msg += (f"🔹 <b>{t['inst']}</b> ({t['team']}) | {t['session']}\n"
                f"الاتجاه: {t['dir']} | Lot: {t['lot']}\n"
                f"الدخول: {t['entry']:.5f} | SL: {t['sl']:.5f}\n-------------------\n")
    send_telegram(msg)

@bot.message_handler(commands=["closeall"])
def close_all_trades(m):
    with STATE_LOCK:
        OPEN_TRADES.clear()
    send_telegram("🚨 تم تصفير قائمة الصفقات المفتوحة محلياً (لا يغلق صفقات حقيقية على وسيط).")

@bot.message_handler(commands=["stats"])
def stats_trades(m):
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("""SELECT COUNT(*),
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END),
                       SUM(profit) FROM trades""")
        res = cur.fetchone()
        cur.close()
        conn.close()
        total, wins, losses, total_profit = res[0] or 0, res[1] or 0, res[2] or 0, res[3] or 0.0
        win_rate = (wins / total * 100) if total else 0.0
        send_telegram(
            f"📈 <b>إحصائيات:</b>\nإجمالي: {total} | 🟢 {wins} | 🔴 {losses}\n"
            f"🎯 النجاح: {win_rate:.1f}% | 💵 الأرباح: ${total_profit:.2f}\n"
            f"عيّنة كافية للتعلّم التكيفي؟ الحد الأدنى {ADAPTIVE_MIN_SAMPLE} صفقة لكل مكوّن."
        )
    except Exception as e:
        send_telegram(f"⚠️ خطأ في جلب الإحصائيات: {e}")

@bot.message_handler(commands=["pause"])
def pause_bot(m):
    global BOT_ACTIVE
    BOT_ACTIVE = False
    send_telegram("⏸️ تم إيقاف المحفظة مؤقتاً.")

@bot.message_handler(commands=["resume"])
def resume_bot(m):
    global BOT_ACTIVE
    BOT_ACTIVE = True
    send_telegram("▶️ تم استئناف العمل.")

@bot.message_handler(commands=["health"])
def health_check(m):
    with HEARTBEAT_LOCK:
        snapshot = dict(LOOP_HEARTBEATS)
    now = datetime.datetime.now(GZA)
    msg = "🩺 <b>صحة النظام:</b>\n"
    for team, last_seen in snapshot.items():
        if last_seen is None:
            msg += f"🔸 {team}: لم يكتمل أول دورة بعد\n"
        else:
            mins = (now - last_seen).total_seconds() / 60
            icon = "🟢" if mins <= HEARTBEAT_STALE_MINUTES else "🔴"
            msg += f"{icon} {team}: آخر نبض منذ {mins:.0f} دقيقة\n"
    send_telegram(msg)


# ========================================================================
# 16. التشغيل الرئيسي
# ========================================================================
if __name__ == "__main__":
    db_setup()

    threading.Thread(target=run_loop, args=("sniper", 10, 45), daemon=True).start()
    threading.Thread(target=run_loop, args=("scalp", 15, 120), daemon=True).start()
    threading.Thread(target=run_loop, args=("daily", 20, 300), daemon=True).start()
    threading.Thread(target=run_loop, args=("swing", 30, 900), daemon=True).start()
    threading.Thread(target=heartbeat_monitor, daemon=True).start()

    log.info("PAL v12.0 Structural Edition يعمل الآن...")
    try:
        bot.infinity_polling()
    except Exception as e:
        log.exception(f"Polling Error: {e}")
