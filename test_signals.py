"""
test_signals.py
================
اختبارات وحدة (unit tests) للدوال الحتمية في signals.py، باستخدام بيانات
شموع صناعية (synthetic OHLC) نبنيها بأنفسنا بحيث نعرف الإجابة الصحيحة مسبقاً.
هذا يحمي من كسر المنطق بصمت عند أي تعديل مستقبلي على signals.py.

التشغيل:
    pip install pytest
    pytest test_signals.py -v
"""

import numpy as np
import pandas as pd
import pytest

from signals import (
    find_swings, market_structure_state, detect_fvg, detect_order_blocks,
    detect_liquidity_sweep, regime_filter, evaluate_signal,
)


# ========================================================================
# دوال مساعدة لبناء بيانات صناعية
# ========================================================================
def make_df(o, h, l, c):
    """يبني DataFrame بأعمدة o/h/l/c من قوائم بسيطة."""
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c})


def make_uptrend_df(n=100, start=100.0, step=0.5, noise=0.1, seed=42):
    """اتجاه صاعد واضح (Higher Highs / Higher Lows) بضوضاء بسيطة."""
    rng = np.random.default_rng(seed)
    closes = start + np.arange(n) * step + rng.normal(0, noise, n)
    opens = closes - rng.normal(0, noise, n)
    highs = np.maximum(opens, closes) + abs(rng.normal(0.2, 0.05, n))
    lows = np.minimum(opens, closes) - abs(rng.normal(0.2, 0.05, n))
    return make_df(opens, highs, lows, closes)


def make_downtrend_df(n=100, start=100.0, step=0.5, noise=0.1, seed=42):
    rng = np.random.default_rng(seed)
    closes = start - np.arange(n) * step + rng.normal(0, noise, n)
    opens = closes + rng.normal(0, noise, n)
    highs = np.maximum(opens, closes) + abs(rng.normal(0.2, 0.05, n))
    lows = np.minimum(opens, closes) - abs(rng.normal(0.2, 0.05, n))
    return make_df(opens, highs, lows, closes)


def make_range_df(n=100, mid=100.0, amplitude=1.0, seed=42):
    """سوق عرضي (نطاق ثابت) بلا اتجاه هيكلي واضح."""
    rng = np.random.default_rng(seed)
    closes = mid + amplitude * np.sin(np.linspace(0, 6 * np.pi, n)) + rng.normal(0, 0.05, n)
    opens = closes - rng.normal(0, 0.05, n)
    highs = np.maximum(opens, closes) + 0.15
    lows = np.minimum(opens, closes) - 0.15
    return make_df(opens, highs, lows, closes)


# ========================================================================
# اختبارات find_swings
# ========================================================================
def test_find_swings_detects_obvious_peak():
    # قمة واضحة عند المنتصف: 1,2,3,10,3,2,1 محاطة بقيم أصغر
    h = [1, 2, 3, 10, 3, 2, 1]
    l = [0.5, 1.5, 2.5, 9.5, 2.5, 1.5, 0.5]
    df = make_df(h, h, l, h)
    result = find_swings(df, window=2)
    assert result["swing_high"].iloc[3] == True  # noqa: E712


def test_find_swings_detects_obvious_trough():
    l = [10, 5, 2, 0.1, 2, 5, 10]
    h = [10.5, 5.5, 2.5, 0.6, 2.5, 5.5, 10.5]
    df = make_df(h, h, l, l)
    result = find_swings(df, window=2)
    assert result["swing_low"].iloc[3] == True  # noqa: E712


# ========================================================================
# اختبارات market_structure_state: الاتجاه الهيكلي
# ========================================================================
def test_structure_detects_uptrend():
    df = make_uptrend_df(n=120)
    state = market_structure_state(df)
    assert state["trend"] == "UPTREND"


def test_structure_detects_downtrend():
    df = make_downtrend_df(n=120)
    state = market_structure_state(df)
    assert state["trend"] == "DOWNTREND"


def test_structure_insufficient_data_returns_unknown():
    df = make_df([1, 2], [1.1, 2.1], [0.9, 1.9], [1, 2])
    state = market_structure_state(df)
    assert state["trend"] == "UNKNOWN"


# ========================================================================
# اختبارات detect_fvg: فجوة السعر العادلة
# ========================================================================
def test_fvg_detects_clear_bullish_gap():
    # شمعة1: high=10 | شمعة2: عادية | شمعة3: low=11 (فجوة واضحة 10% > الحد الأدنى)
    o = [9, 10.5, 11.5]
    h = [10, 11, 12]
    l = [8.5, 10, 11]
    c = [9.5, 10.8, 11.8]
    df = make_df(o, h, l, c)
    gaps = detect_fvg(df, min_gap_pct=0.001)
    assert len(gaps) == 1
    assert gaps[0]["type"] == "BULLISH"


def test_fvg_detects_clear_bearish_gap():
    o = [12, 10.5, 9]
    h = [12.5, 11, 9.5]
    l = [11, 9.5, 8]
    c = [11.5, 10, 8.5]
    df = make_df(o, h, l, c)
    gaps = detect_fvg(df, min_gap_pct=0.001)
    assert len(gaps) == 1
    assert gaps[0]["type"] == "BEARISH"


def test_fvg_no_gap_when_candles_overlap():
    # شموع متراكبة تماماً - لا فجوة إطلاقاً
    o = [10, 10, 10]
    h = [11, 11, 11]
    l = [9, 9, 9]
    c = [10, 10, 10]
    df = make_df(o, h, l, c)
    gaps = detect_fvg(df, min_gap_pct=0.001)
    assert len(gaps) == 0


# ========================================================================
# اختبارات detect_liquidity_sweep: القمم/القيعان المتساوية
# ========================================================================
def test_liquidity_sweep_detects_equal_highs():
    # قمتان متقاربتان جداً (ضمن سماحية 0.08%) بفاصل زمني كافٍ لاكتشافهما كـ swings
    n = 40
    h = [100.0] * n
    l = [95.0] * n
    # نضع قمتين بارزتين متساويتين تقريباً عند مواضع متباعدة
    h[10] = 110.0
    h[11] = 109.0
    h[9] = 109.0
    h[30] = 110.05  # قريبة جداً من القمة الأولى (0.05% فرق)
    h[29] = 109.0
    h[31] = 109.0
    o = h.copy()
    c = h.copy()
    l2 = [x - 5 for x in h]
    df = make_df(o, h, l2, c)
    sweeps = detect_liquidity_sweep(df, tolerance_pct=0.005)
    types = [s["type"] for s in sweeps]
    assert "EQUAL_HIGHS" in types


# ========================================================================
# اختبارات regime_filter
# ========================================================================
def test_regime_filter_unknown_with_insufficient_data():
    df = make_df([1, 2], [1.1, 2.1], [0.9, 1.9], [1, 2])
    assert regime_filter(df) == "UNKNOWN"


def test_regime_filter_detects_trending_market():
    df = make_uptrend_df(n=100, step=1.0, noise=0.05)  # اتجاه قوي وواضح
    regime = regime_filter(df)
    assert regime in ("TRENDING", "TRANSITIONAL")  # ADX يحتاج بيانات كافية ونظيفة


# ========================================================================
# اختبارات evaluate_signal: القرار المجمّع
# ========================================================================
def test_evaluate_signal_returns_none_direction_with_flat_data():
    """بيانات شبه مسطحة بلا أي إشارة واضحة يجب ألا تُنتج قراراً بثقة زائفة."""
    df = make_df([100] * 60, [100.2] * 60, [99.8] * 60, [100] * 60)
    decision = evaluate_signal(df, df, base_threshold=3)
    assert decision["direction"] is None


def test_evaluate_signal_structure_only_weight_isolates_component():
    """عند تصفير وزن smc وmomentum، يجب ألا يتأثر total بهما إطلاقاً."""
    df = make_uptrend_df(n=120)
    weights_full = {"structure": 1.0, "smc": 1.0, "momentum": 1.0}
    weights_struct_only = {"structure": 1.0, "smc": 0.0, "momentum": 0.0}
    d_full = evaluate_signal(df, df, base_threshold=1, weights=weights_full)
    d_struct = evaluate_signal(df, df, base_threshold=1, weights=weights_struct_only)
    expected = d_full["struct_score"] * 1.0
    assert d_struct["total"] == pytest.approx(expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
