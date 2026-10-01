"""
05_zero_consumption.py
Module for detecting Prolonged Zero Consumption Anomaly while telemetry packet transmission continues.
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_zero_consumption(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    min_zero_days: int = 90
) -> dict:
    """
    Identifies devices with continuous zero consumption (daily_flow == 0)
    for at least min_zero_days consecutive days while active packet transmission continues (paketov_za_sutki >= 1).
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Filter to valid daily records sending at least 1 telemetry packet
    active_daily = df_pok[(df_pok["date_diff"] == 1) & (df_pok["paketov_za_sutki"] >= 1)].copy()
    active_daily["is_zero"] = (active_daily["daily_flow"].abs() < 1e-6).astype(int)

    def calc_max_zero_streak(df_dev):
        s = df_dev["is_zero"]
        blocks = (s != s.shift()).cumsum()
        streaks = s.groupby(blocks).transform("sum") * s
        max_streak = streaks.max() if len(streaks) > 0 else 0
        total_zero_days = s.sum()
        total_packets = df_dev["paketov_za_sutki"].sum()
        return pd.Series({
            "max_zero_streak": int(max_streak),
            "total_zero_days": int(total_zero_days),
            "total_packets_sent": int(total_packets)
        })

    streak_df = active_daily.groupby("anon_id").apply(calc_max_zero_streak, include_groups=False).reset_index()

    zero_devices = streak_df[streak_df["max_zero_streak"] >= min_zero_days].copy()
    zero_devices = zero_devices.sort_values("max_zero_streak", ascending=False).reset_index(drop=True)
    zero_devices = zero_devices.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    top_examples = zero_devices.head(5).to_dict(orient="records")

    threshold_counts = {
        ">= 30 days": int((streak_df["max_zero_streak"] >= 30).sum()),
        ">= 60 days": int((streak_df["max_zero_streak"] >= 60).sum()),
        ">= 90 days": int((streak_df["max_zero_streak"] >= 90).sum()),
        ">= 180 days": int((streak_df["max_zero_streak"] >= 180).sum()),
        ">= 300 days": int((streak_df["max_zero_streak"] >= 300).sum())
    }

    result = {
        "anomaly_name": "Длительное нулевое потребление (Prolonged Zero Consumption)",
        "criterion": f"Нулевое потребление (daily_flow = 0) при активной телеметрии (paketov_za_sutki >= 1) подряд в течение >= {min_zero_days} дней",
        "threshold_days": min_zero_days,
        "total_affected_devices": int(len(zero_devices)),
        "affected_device_ids": zero_devices["anon_id"].tolist(),
        "threshold_counts": threshold_counts,
        "device_summary": zero_devices,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины длительного нулевого потребления:\n"
            "1. Нежилая / незаселенная квартира (собственники не проживают, уехали в долгосрочную отпуск/командировку).\n"
            "2. Заклинивание счетного механизма ('застрявший счетчик'): электронный контроллер исправно слает телеметрию, "
            "но механическая крыльчатка заблокирована окалиной, инородным телом или механическим износом.\n"
            "3. Умышленное несанкционированное вмешательство (блокировка крыльчатки иголками/магнитами)."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_zero_consumption()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов (>= 90 дней): {res['total_affected_devices']}")
    print(f"Распределение по порогам: {res['threshold_counts']}")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | Дней нулевого расхода подряд: {ex['max_zero_streak']} | Передано пакетов: {ex['total_packets_sent']} | {ex['tip']}")
