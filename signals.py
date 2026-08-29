"""
signals.py
==========
منطق التحليل الحتمي (بنية السوق + SMC/ICT + الزخم + فلتر النظام) بمعزل تام
عن Telegram وقاعدة البيانات وأي حالة تشغيل (state) عالمية.

سبب الفصل: bot.py يُشغّل TeleBot و psycopg2 فور استيراده (import)، ما يمنع
استخدام نفس دوال التحليل داخل backtester.py أو test_signals.py بدون تشغيل
بوت حقيقي أو الاتصال بقاعدة بيانات فعلية. هذا الملف يحتوي فقط دوالاً نقية
(pure functions): تُدخل DataFrame وتُخرج نتيجة، بلا أي أثر جانبي.

bot.py و backtester.py و test_signals.py الثلاثة تستورد من هنا.
"""

import numpy as np
import pandas as pd
from ta import trend, momentum
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator


# ========================================================================
# بنية السوق: Swing Points, BOS/CHoCH
# ========================================================================
def find_swings(df, window=3):
    """اكتشاف القمم والقيعان (fractal-based). يُرجع أعمدة swing_high/swing_low."""
    highs = df["h"].values
    lows = df["l"].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_high[i] = True
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_low[i] = True
    df = df.copy()
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def market_structure_state(df):
    """يحدد الاتجاه الهيكلي (HH/HL | LH/LL | Range) وآخر حدث BOS/CHoCH."""
    df = find_swings(df)
    highs = df[df["swing_high"]]["h"]
    lows = df[df["swing_low"]]["l"]

    if len(highs) < 2 or len(lows) < 2:
        return {"trend": "UNKNOWN", "event": None, "last_high": None, "last_low": None}

    last_two_highs = highs.tail(2).values
    last_two_lows = lows.tail(2).values

    higher_high = last_two_highs[-1] > last_two_highs[-2]
    higher_low = last_two_lows[-1] > last_two_lows[-2]
    lower_high = last_two_highs[-1] < last_two_highs[-2]
    lower_low = last_two_lows[-1] < last_two_lows[-2]

    if higher_high and higher_low:
        struct_trend = "UPTREND"
    elif lower_high and lower_low:
        struct_trend = "DOWNTREND"
    else:
        struct_trend = "RANGE"

    current_price = df["c"].iloc[-1]
    event = None
    if struct_trend == "UPTREND" and current_price < last_two_lows[-2]:
        event = "CHoCH_BEARISH"
    elif struct_trend == "DOWNTREND" and current_price > last_two_highs[-2]:
        event = "CHoCH_BULLISH"
    elif struct_trend == "UPTREND" and current_price > last_two_highs[-1]:
        event = "BOS_BULLISH"
    elif struct_trend == "DOWNTREND" and current_price < last_two_lows[-1]:
        event = "BOS_BEARISH"

    return {
        "trend": struct_trend,
        "event": event,
        "last_high": float(last_two_highs[-1]),
        "last_low": float(last_two_lows[-1]),
    }


def get_dynamic_sr_zones(df, tolerance_pct=0.0015):
    """تجميع القمم/القيعان القريبة كمناطق دعم/مقاومة (وليس خطوطاً جامدة)."""
    df = find_swings(df)
    levels = pd.concat([df[df["swing_high"]]["h"], df[df["swing_low"]]["l"]]).sort_values()
    zones = []
    for lvl in levels:
        placed = False
        for z in zones:
            if abs(lvl - z["mid"]) / z["mid"] <= tolerance_pct:
                z["touches"] += 1
                z["mid"] = (z["mid"] * (z["touches"] - 1) + lvl) / z["touches"]
                placed = True
                break
        if not placed:
            zones.append({"mid": float(lvl), "touches": 1})
    zones.sort(key=lambda z: z["touches"], reverse=True)
    return zones[:6]


def regime_filter(df):
    """يحدد هل السوق متجه بقوة (TRENDING) أم عرضي (RANGING) عبر ADX."""
    if len(df) < 30:
        return "UNKNOWN"
    adx = ADXIndicator(high=df["h"], low=df["l"], close=df["c"], window=14)
    adx_val = adx.adx().iloc[-1]
    if pd.isna(adx_val):
        return "UNKNOWN"
    if adx_val >= 25:
        return "TRENDING"
    elif adx_val <= 18:
        return "RANGING"
    return "TRANSITIONAL"


# ========================================================================
# SMC / ICT: Order Blocks, FVG, Liquidity Sweeps
# ========================================================================
def detect_fvg(df, min_gap_pct=0.0005):
    """Fair Value Gap: فجوة بين شمعة1 وشمعة3 لا تغطيها شمعة2."""
    gaps = []
    h = df["h"].values
    l = df["l"].values
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            gap_size = (l[i] - h[i - 2]) / h[i - 2]
            if gap_size >= min_gap_pct:
                gaps.append({"type": "BULLISH", "top": float(l[i]), "bottom": float(h[i - 2]), "idx": i})
        elif h[i] < l[i - 2]:
            gap_size = (l[i - 2] - h[i]) / l[i - 2]
            if gap_size >= min_gap_pct:
                gaps.append({"type": "BEARISH", "top": float(l[i - 2]), "bottom": float(h[i]), "idx": i})
    return gaps[-5:]


def detect_order_blocks(df, impulse_atr_mult=1.5):
    """Order Block: آخر شمعة معاكسة قبل حركة اندفاعية (>1.5×ATR)."""
    if len(df) < 20:
        return []
    atr = AverageTrueRange(high=df["h"], low=df["l"], close=df["c"], window=14).average_true_range()
    obs = []
    for i in range(15, len(df) - 1):
        if atr.iloc[i] == 0 or np.isnan(atr.iloc[i]):
            continue
        candle_range = df["h"].iloc[i] - df["l"].iloc[i]
        if candle_range >= impulse_atr_mult * atr.iloc[i]:
            is_bullish_impulse = df["c"].iloc[i] > df["o"].iloc[i]
            prev_idx = i - 1
            prev_is_opposite = (
                (df["c"].iloc[prev_idx] < df["o"].iloc[prev_idx]) if is_bullish_impulse
                else (df["c"].iloc[prev_idx] > df["o"].iloc[prev_idx])
            )
            if prev_is_opposite:
                obs.append({
                    "type": "BULLISH_OB" if is_bullish_impulse else "BEARISH_OB",
                    "top": float(max(df["o"].iloc[prev_idx], df["c"].iloc[prev_idx])),
                    "bottom": float(min(df["o"].iloc[prev_idx], df["c"].iloc[prev_idx])),
                    "idx": prev_idx,
                })
    return obs[-5:]


def detect_liquidity_sweep(df, tolerance_pct=0.0008):
    """Equal Highs/Lows - مستويات تجمّع أوامر وقف الخسارة المفترضة."""
    df = find_swings(df)
    highs = df[df["swing_high"]][["h"]].tail(6)
    lows = df[df["swing_low"]][["l"]].tail(6)
    sweeps = []
    for i in range(len(highs) - 1):
        h1, h2 = highs["h"].iloc[i], highs["h"].iloc[i + 1]
        if abs(h1 - h2) / h1 <= tolerance_pct:
            sweeps.append({"type": "EQUAL_HIGHS", "level": float(max(h1, h2))})
    for i in range(len(lows) - 1):
        l1, l2 = lows["l"].iloc[i], lows["l"].iloc[i + 1]
        if abs(l1 - l2) / l1 <= tolerance_pct:
            sweeps.append({"type": "EQUAL_LOWS", "level": float(min(l1, l2))})
    return sweeps


# ========================================================================
# موجات إليوت - معلوماتي فقط
# ========================================================================
def elliott_wave_context(df):
    """عدّاد موجات مبسّط بقواعد Elliott الصارمة. معلوماتي فقط - لا يدخل بالقرار."""
    df = find_swings(df, window=2)
    pivots = df[df["swing_high"] | df["swing_low"]].tail(6)
    if len(pivots) < 5:
        return {"count": "INSUFFICIENT_DATA", "valid": False}

    prices = pivots["h"].where(pivots["swing_high"], pivots["l"]).values
    if len(prices) < 5:
        return {"count": "INSUFFICIENT_DATA", "valid": False}

    w1 = abs(prices[1] - prices[0])
    w2 = abs(prices[2] - prices[1])
    w3 = abs(prices[3] - prices[2])
    w5 = abs(prices[4] - prices[3]) if len(prices) > 4 else None

    wave2_valid = w2 < w1
    wave3_valid = True
    if w5 is not None:
        wave3_valid = not (w3 < w1 and w3 < w5)

    return {
        "count": "POSSIBLE_IMPULSE" if (wave2_valid and wave3_valid) else "RULES_VIOLATED",
        "valid": wave2_valid and wave3_valid,
    }


# ========================================================================
# مكونات الإشارة الثلاثة المستقلة
# ========================================================================
def structure_signal(df):
    state = market_structure_state(df)
    score = 0
    if state["event"] == "BOS_BULLISH":
        score = 2
    elif state["event"] == "BOS_BEARISH":
        score = -2
    elif state["event"] == "CHoCH_BULLISH":
        score = 1
    elif state["event"] == "CHoCH_BEARISH":
        score = -1
    label = f"البنية: {state['trend']} | الحدث: {state['event'] or 'لا يوجد'}"
    return score, label, state


def smc_signal(df):
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df)
    score = 0
    parts = []
    if fvgs:
        last_fvg = fvgs[-1]
        score += 1 if last_fvg["type"] == "BULLISH" else -1
        parts.append(f"FVG أخير: {last_fvg['type']}")
    if obs:
        last_ob = obs[-1]
        score += 1 if last_ob["type"] == "BULLISH_OB" else -1
        parts.append(f"OB أخير: {last_ob['type']}")
    label = " | ".join(parts) if parts else "SMC: لا توجد إشارة واضحة"
    return score, label, {"fvgs": fvgs, "obs": obs}


def momentum_signal(df):
    if len(df) < 20:
        return 0, "الزخم: بيانات ناقصة", {}
    rsi = momentum.RSIIndicator(df["c"], window=14).rsi().iloc[-1]
    macd = trend.MACD(df["c"])
    macd_diff = macd.macd_diff().iloc[-1]
    if pd.isna(rsi) or pd.isna(macd_diff):
        return 0, "الزخم: بيانات ناقصة", {}
    score = 0
    if rsi < 30 and macd_diff > 0:
        score = 2
    elif rsi > 70 and macd_diff < 0:
        score = -2
    elif macd_diff > 0:
        score = 1
    elif macd_diff < 0:
        score = -1
    label = f"RSI: {rsi:.1f} | MACD Diff: {macd_diff:.5f}"
    return score, label, {"rsi": rsi, "macd_diff": macd_diff}


# ========================================================================
# دالة القرار المجمّعة - نفس منطق King_Brain لكن بلا أي أثر جانبي
# (لا Telegram، لا DB، لا OPEN_TRADES) - يستخدمها bot.py وbacktester.py معاً
# ========================================================================
def evaluate_signal(df_trend, df_entry, base_threshold, weights=None):
    """
    يُرجع dict فيه: direction (BUY/SELL/None)، total، threshold الفعلي بعد
    فلتر النظام، ومكونات الإشارة الثلاثة. لا يحسب lot/entry/sl/tp - هذه
    مسؤولية الطبقة التي تستدعي (bot.py أو backtester.py) لأنها تختلف حسب
    السياق (تنفيذ حي مقابل محاكاة تاريخية).
    """
    weights = weights or {"structure": 1.0, "smc": 1.0, "momentum": 1.0}

    regime = regime_filter(df_trend)
    threshold = base_threshold
    if regime == "RANGING":
        threshold += 1

    struct_score, struct_label, struct_state = structure_signal(df_trend)
    smc_score, smc_label, smc_data = smc_signal(df_entry)
    mom_score, mom_label, mom_data = momentum_signal(df_entry)

    if regime == "TRENDING" and struct_state["trend"] in ("UPTREND", "DOWNTREND"):
        if (struct_state["trend"] == "UPTREND" and smc_score < 0) or \
           (struct_state["trend"] == "DOWNTREND" and smc_score > 0):
            smc_score = 0

    total = (
        struct_score * weights["structure"]
        + smc_score * weights["smc"]
        + mom_score * weights["momentum"]
    )

    direction = None
    if total >= threshold:
        direction = "BUY"
    elif total <= -threshold:
        direction = "SELL"

    return {
        "direction": direction,
        "total": total,
        "threshold": threshold,
        "regime": regime,
        "struct_score": struct_score, "struct_label": struct_label, "struct_state": struct_state,
        "smc_score": smc_score, "smc_label": smc_label, "smc_data": smc_data,
        "mom_score": mom_score, "mom_label": mom_label, "mom_data": mom_data,
    }
