"""
04_leakage.py
Module for detecting Leakage Anomaly (Continuous water flow without any zero-flow days).
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_leakage(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    min_continuous_days: int = 90
) -> dict:
    """
    Identifies devices with continuous non-zero water flow (daily_flow > 0.001 m³/day)
    for at least min_continuous_days consecutive days (default 90 days).
    
    Indicates a permanent leak (running toilet, dripping tap, riser pipe leak) or commercial continuous operation.
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Filter to consecutive days
    consec = df_pok[df_pok["date_diff"] == 1].copy()
    consec["is_positive"] = (consec["daily_flow"] > 0.001).astype(int)

    # Function to extract max non-zero streak length per device
    def calc_max_positive_streak(df_dev):
        s = df_dev["is_positive"]
        blocks = (s != s.shift()).cumsum()
        streaks = s.groupby(blocks).transform("sum") * s
        max_streak = streaks.max() if len(streaks) > 0 else 0
        total_pos_days = s.sum()
        total_days = len(s)
        avg_flow = df_dev[df_dev["daily_flow"] > 0.001]["daily_flow"].mean() if total_pos_days > 0 else 0.0
        return pd.Series({
            "max_positive_streak": int(max_streak),
            "total_positive_days": int(total_pos_days),
            "total_monitored_days": int(total_days),
            "avg_leak_flow": float(avg_flow)
        })

    streak_df = consec.groupby("anon_id").apply(calc_max_positive_streak, include_groups=False).reset_index()

    # Filter devices exceeding threshold
    leak_devices = streak_df[streak_df["max_positive_streak"] >= min_continuous_days].copy()
    leak_devices = leak_devices.sort_values("max_positive_streak", ascending=False).reset_index(drop=True)
    leak_devices = leak_devices.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    top_examples = leak_devices.head(5).to_dict(orient="records")

    # Multi-threshold statistics for context
    threshold_counts = {
        ">= 30 days": int((streak_df["max_positive_streak"] >= 30).sum()),
        ">= 60 days": int((streak_df["max_positive_streak"] >= 60).sum()),
        ">= 90 days": int((streak_df["max_positive_streak"] >= 90).sum()),
        ">= 180 days": int((streak_df["max_positive_streak"] >= 180).sum()),
        "365 days (Full Year)": int((streak_df["max_positive_streak"] >= 365).sum())
    }

    result = {
        "anomaly_name": "Утечка (Continuous Leakage)",
        "criterion": f"Непрерывное потребление (daily_flow > 0.001 м³/сут) подряд в течение >= {min_continuous_days} дней без «нулевых» суток",
        "threshold_days": min_continuous_days,
        "total_affected_devices": int(len(leak_devices)),
        "affected_device_ids": leak_devices["anon_id"].tolist(),
        "threshold_counts": threshold_counts,
        "device_summary": leak_devices,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины непрерывного потребления (утечки):\n"
            "1. Постоянная протечка сантехнической арматуры (протекающий бачок унитаза, капающий кран, неисправный клапан).\n"
            "2. Микротрещина или свищ во внутриквартирной разводке трубы.\n"
            "3. Коммерческое непрерывное использование (круглосуточные объекты, серверные с водяным охлаждением, общепит)."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_leakage()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов (>= 90 дней): {res['total_affected_devices']}")
    print(f"Распределение по порогам: {res['threshold_counts']}")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | Дней без отдыха: {ex['max_positive_streak']} | Средний расход при утечке: {ex['avg_leak_flow']:.3f} м³/сут | {ex['tip']}")
