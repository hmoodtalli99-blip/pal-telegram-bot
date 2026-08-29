"""
backtester.py
==============
اختبار تاريخي (Backtesting) + تحقق تدريجي (Walk-Forward) على بيانات Finnhub
التاريخية، باستخدام نفس منطق القرار الموجود في signals.py بالحرف - لا نسخة
مختلفة أو مبسّطة، لتفادي أشهر خطأ في الـ backtesting: أن تختبر منطقاً غير
الذي يعمل فعلياً في الإنتاج.

يقيس لكل مكوّن على حدة (structure/smc/momentum) ولمجموعها:
- عدد الصفقات، نسبة الفوز
- Profit Factor (مجموع الأرباح / مجموع الخسائر المطلقة)
- Max Drawdown (أقصى تراجع من قمة الرصيد)

الاستخدام:
    python backtester.py --symbol XAUUSD --timeframe 60 --entry_tf 5 --days 180

ملاحظة: هذا سكربت يُشغَّل يدوياً محلياً أو عبر CI، وليس جزءاً من bot.py
المنشور على Railway - لا يحتاج TELEGRAM_TOKEN ولا DATABASE_URL.
"""

import os
import argparse
import datetime

import requests
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

from signals import evaluate_signal


FINNHUB_KEY = os.environ.get("FINNHUB_KEY")


# ========================================================================
# جلب بيانات تاريخية أعمق مما يحتاجه التشغيل الحي (limit كبير + مدى زمني)
# ========================================================================
def fetch_historical(symbol, resolution, days):
    if not FINNHUB_KEY:
        raise SystemExit("FINNHUB_KEY غير موجود في متغيرات البيئة. صدّره أولاً:\n"
                          "  export FINNHUB_KEY=xxxx")
    to_ts = int(datetime.datetime.now().timestamp())
    from_ts = to_ts - days * 24 * 3600

    if symbol == "XAUUSD":
        url = (f"https://api.finnhub.io/api/v1/crypto/candle?symbol=BINANCE:XAUUSD"
               f"&resolution={resolution}&from={from_ts}&to={to_ts}&token={FINNHUB_KEY}")
    else:
        url = (f"https://api.finnhub.io/api/v1/forex/candle?symbol=OANDA:{symbol}"
               f"&resolution={resolution}&from={from_ts}&to={to_ts}&token={FINNHUB_KEY}")

    res = requests.get(url, timeout=15)
    data = res.json()
    if data.get("s") != "ok" or not data.get("c"):
        raise SystemExit(f"فشل جلب البيانات التاريخية لـ {symbol}: {data}")

    df = pd.DataFrame({
        "o": data["o"], "h": data["h"], "l": data["l"], "c": data["c"],
        "t": pd.to_datetime(data["t"], unit="s"),
    })
    return df


# ========================================================================
# محاكاة صفقة واحدة: من نقطة الإشارة حتى TP أو SL أو نهاية البيانات
# (تبسيط متعمّد: لا محاكاة لتأمين الوقف/الهدف الجزئي كما في bot.py الحي،
#  لأن الهدف هنا قياس جودة الإشارة الخام قبل طبقات إدارة الصفقة)
# ========================================================================
def simulate_trade(df, entry_idx, direction, entry, sl, tp, max_bars_ahead=200):
    future = df.iloc[entry_idx + 1: entry_idx + 1 + max_bars_ahead]
    for _, row in future.iterrows():
        if direction == "BUY":
            if row["l"] <= sl:
                return "LOSS", sl - entry
            if row["h"] >= tp:
                return "WIN", tp - entry
        else:
            if row["h"] >= sl:
                return "LOSS", entry - sl
            if row["l"] <= tp:
                return "WIN", entry - tp
    return "OPEN_AT_END", 0.0  # لم يُغلق ضمن نافذة المحاكاة - يُستبعد من الإحصاء


# ========================================================================
# تشغيل backtest كامل على مكوّن واحد فقط (لعزل جودة كل مصدر إشارة)
# ========================================================================
def run_backtest(df_trend, df_entry, weights, base_threshold=3, tp_mul=2.0,
                  min_bars=60, step=1):
    """
    يمشي عبر البيانات شمعة بشمعة (بعد نافذة إحماء min_bars)، يستدعي
    evaluate_signal بالضبط كما يفعل bot.py، ويحاكي نتيجة كل إشارة.
    """
    trades = []
    n = min(len(df_trend), len(df_entry))

    for i in range(min_bars, n - 1, step):
        window_trend = df_trend.iloc[max(0, i - 200): i + 1].reset_index(drop=True)
        window_entry = df_entry.iloc[max(0, i - 200): i + 1].reset_index(drop=True)
        if len(window_trend) < 50 or len(window_entry) < 50:
            continue

        decision = evaluate_signal(window_trend, window_entry, base_threshold, weights=weights)
        if decision["direction"] is None:
            continue

        atr = AverageTrueRange(
            high=window_entry["h"], low=window_entry["l"], close=window_entry["c"], window=14
        ).average_true_range().iloc[-1]
        if pd.isna(atr) or atr == 0:
            continue

        entry_price = window_entry["c"].iloc[-1]
        if decision["direction"] == "BUY":
            sl = entry_price - atr * 1.2
            tp = entry_price + atr * 1.2 * tp_mul
        else:
            sl = entry_price + atr * 1.2
            tp = entry_price - atr * 1.2 * tp_mul

        result, pnl_points = simulate_trade(df_entry, i, decision["direction"], entry_price, sl, tp)
        if result == "OPEN_AT_END":
            continue

        trades.append({
            "idx": i, "direction": decision["direction"], "result": result,
            "pnl": pnl_points, "regime": decision["regime"],
        })

    return trades


# ========================================================================
# مقاييس الأداء
# ========================================================================
def compute_metrics(trades):
    if not trades:
        return {"count": 0, "win_rate": None, "profit_factor": None, "max_drawdown": None}

    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.cumsum([t["pnl"] for t in trades])
    running_max = np.maximum.accumulate(equity) if len(equity) else np.array([0])
    drawdown = running_max - equity
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


# ========================================================================
# Walk-Forward: تقسيم البيانات لفترات تدريب/اختبار متعاقبة
# (هنا "تدريب" يعني فقط قياس الأداء على فترة، لأن أوزان signals.py ثابتة
#  حالياً - عند تفعيل meta-model حقيقي لاحقاً، هذا هو المكان الذي يعاد فيه
#  ضبط الأوزان على نافذة التدريب قبل اختبارها على نافذة لم يرها النظام)
# ========================================================================
def walk_forward_split(df_trend, df_entry, n_folds=4):
    n = min(len(df_trend), len(df_entry))
    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = n if f == n_folds - 1 else (f + 1) * fold_size
        folds.append((start, end))
    return folds


def run_walk_forward(df_trend, df_entry, weights, base_threshold=3, n_folds=4):
    folds = walk_forward_split(df_trend, df_entry, n_folds)
    results = []
    for i, (start, end) in enumerate(folds):
        seg_trend = df_trend.iloc[start:end].reset_index(drop=True)
        seg_entry = df_entry.iloc[start:end].reset_index(drop=True)
        trades = run_backtest(seg_trend, seg_entry, weights, base_threshold)
        metrics = compute_metrics(trades)
        results.append({"fold": i + 1, **metrics})
    return results


# ========================================================================
# نقطة الدخول: يختبر كل مكوّن منفرداً + المجموع، ثم walk-forward للمجموع
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description="PAL Backtester")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="60", help="فريم الاتجاه (دقائق أو D)")
    parser.add_argument("--entry_tf", default="15", help="فريم الدخول")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    print(f"جلب بيانات {args.symbol} لآخر {args.days} يوماً...")
    df_trend = fetch_historical(args.symbol, args.timeframe, args.days)
    df_entry = fetch_historical(args.symbol, args.entry_tf, args.days)
    print(f"شموع الاتجاه: {len(df_trend)} | شموع الدخول: {len(df_entry)}")

    component_weights = {
        "الكل مجتمعاً": {"structure": 1.0, "smc": 1.0, "momentum": 1.0},
        "structure فقط": {"structure": 1.0, "smc": 0.0, "momentum": 0.0},
        "smc فقط": {"structure": 0.0, "smc": 1.0, "momentum": 0.0},
        "momentum فقط": {"structure": 0.0, "smc": 0.0, "momentum": 1.0},
    }

    print("\n" + "=" * 60)
    print(f"{'المكوّن':<20}{'عدد الصفقات':<14}{'نسبة الفوز':<14}{'Profit Factor':<16}{'أقصى تراجع':<12}")
    print("=" * 60)
    for label, weights in component_weights.items():
        # عند اختبار مكوّن منفرد، العتبة الفعالة تتقلص لأن أقصى مجموع ممكن هو 2
        eff_threshold = 2 if sum(1 for w in weights.values() if w > 0) == 1 else args.threshold
        trades = run_backtest(df_trend, df_entry, weights, base_threshold=eff_threshold)
        m = compute_metrics(trades)
        wr = f"{m['win_rate']:.1f}%" if m["win_rate"] is not None else "-"
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] not in (None, float("inf")) else "-"
        dd = f"{m['max_drawdown']:.5f}" if m["max_drawdown"] is not None else "-"
        print(f"{label:<20}{m['count']:<14}{wr:<14}{pf:<16}{dd:<12}")

    print("\n" + "=" * 60)
    print(f"Walk-Forward ({args.folds} فترات) - الإشارة المجمّعة كاملة")
    print("=" * 60)
    wf_results = run_walk_forward(df_trend, df_entry, component_weights["الكل مجتمعاً"], args.threshold, args.folds)
    for r in wf_results:
        wr = f"{r['win_rate']:.1f}%" if r["win_rate"] is not None else "-"
        pf = f"{r['profit_factor']:.2f}" if r["profit_factor"] not in (None, float("inf")) else "-"
        print(f"فترة {r['fold']}: صفقات={r['count']} | فوز={wr} | PF={pf} | تراجع={r['max_drawdown']}")

    print("\nملاحظة: نتيجة متذبذبة بشدة بين الفترات (fold) تشير لعدم استقرار")
    print("الاستراتيجية عبر الزمن (احتمال overfitting) وليس بالضرورة خللاً في الكود.")


if __name__ == "__main__":
    main()
