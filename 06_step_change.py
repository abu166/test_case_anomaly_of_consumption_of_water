"""
06_step_change.py
Module for detecting Step Change Anomaly (Sharp, sustained shift in baseline daily water consumption).
"""

import pandas as pd
import numpy as np
from data_loader import load_pokazaniya, load_pribory


def analyze_step_change(
    df_pok: pd.DataFrame = None,
    df_prib: pd.DataFrame = None,
    split_date: str = "2026-04-01",
    min_days_per_period: int = 30,
    min_ratio: float = 3.0,
    min_abs_diff: float = 0.15
) -> dict:
    """
    Identifies devices with a sustained shift in consumption level between first and second half of monitoring period.
    
    Excludes corrupted telemetry outliers (> 50 m³/day) from mean calculations.
    Criteria:
    - Period 1: before split_date, Period 2: on/after split_date
    - Ratio: Mean(Period 2) / Mean(Period 1) >= min_ratio OR <= (1 / min_ratio)
    - Absolute shift: |Mean(Period 2) - Mean(Period 1)| >= min_abs_diff m³/day
    """
    if df_pok is None:
        df_pok = load_pokazaniya()
    if df_prib is None:
        df_prib = load_pribory()

    # Clean out extreme telemetry noise > 50 m³
    clean_pok = df_pok[(df_pok["date_diff"] == 1) & (df_pok["daily_flow"] >= 0) & (df_pok["daily_flow"] <= 50.0)].copy()

    split_dt = pd.to_datetime(split_date)

    p1 = clean_pok[clean_pok["data"] < split_dt].groupby("anon_id")["daily_flow"].agg(
        mean1="mean", median1="median", count1="count"
    )
    p2 = clean_pok[clean_pok["data"] >= split_dt].groupby("anon_id")["daily_flow"].agg(
        mean2="mean", median2="median", count2="count"
    )

    step_df = p1.join(p2, how="inner")
    step_df = step_df[(step_df["count1"] >= min_days_per_period) & (step_df["count2"] >= min_days_per_period)].copy()

    # Calculate step metrics
    step_df["ratio"] = (step_df["mean2"] + 0.001) / (step_df["mean1"] + 0.001)
    step_df["abs_diff"] = (step_df["mean2"] - step_df["mean1"]).abs()

    step_up_mask = (step_df["ratio"] >= min_ratio) & (step_df["abs_diff"] >= min_abs_diff)
    step_down_mask = (step_df["ratio"] <= (1.0 / min_ratio)) & (step_df["abs_diff"] >= min_abs_diff)

    step_df["step_type"] = np.where(step_up_mask, "Резкий рост (Step Up)", np.where(step_down_mask, "Резкое падение (Step Down)", "Normal"))

    affected_df = step_df[step_df["step_type"] != "Normal"].reset_index()
    affected_df = affected_df.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    affected_df = affected_df.merge(df_prib[["anon_id", "tip", "model"]], on="anon_id", how="left")

    top_examples = affected_df.head(5).to_dict(orient="records")

    step_up_count = int(step_up_mask.sum())
    step_down_count = int(step_down_mask.sum())

    result = {
        "anomaly_name": "Ступенька (Step Change / Baseline Shift)",
        "criterion": (
            f"Резкое изменение среднего суточного расхода между периодами (до и после {split_date}): "
            f"Отношение средних >= {min_ratio}x (или <= {1/min_ratio:.2f}x) И абсолютный сдвиг >= {min_abs_diff} м³/сут"
        ),
        "split_date": split_date,
        "step_up_count": step_up_count,
        "step_down_count": step_down_count,
        "total_affected_devices": int(len(affected_df)),
        "affected_device_ids": affected_df["anon_id"].tolist(),
        "device_summary": affected_df,
        "top_examples": top_examples,
        "causes_explanation": (
            "Основные причины ступенчатого изменения расхода:\n"
            "1. Изменение состава проживающих (заселение арендаторов, рождение ребенка, сдача квартиры в аренду).\n"
            "2. Ввод в эксплуатацию нового моющего/климатического оборудования (стиральная/посудомоечная машина, система очистки воды).\n"
            "3. Образование скрытой постоянной утечки (скачок вверх) или устранение постоянной утечки (скачок вниз).\n"
            "4. Частичное заклинивание или повреждение механического узла счетчика."
        )
    }
    return result


if __name__ == "__main__":
    res = analyze_step_change()
    print(f"=== {res['anomaly_name']} ===")
    print(f"Критерий: {res['criterion']}")
    print(f"Количество найденных приборов: {res['total_affected_devices']} (Рост: {res['step_up_count']}, Падение: {res['step_down_count']})")
    print("\nПримеры приборов:")
    for ex in res["top_examples"]:
        print(f"  - anon_id: {ex['anon_id']} | {ex['step_type']} | До: {ex['mean1']:.3f} м³/сут -> После: {ex['mean2']:.3f} м³/сут (Кратность: {ex['ratio']:.1f}x) | {ex['tip']}")
