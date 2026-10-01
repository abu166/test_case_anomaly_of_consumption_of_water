"""
generate_report.py
Master report generator for Water Consumption Anomaly Detection Test Assignment.
Executes all 8 analytical modules, aggregates findings, and produces a structured Markdown report.
"""

import sys
import os
import pandas as pd
import numpy as np

from data_loader import load_full_dataset
from importlib import import_module

# Import analysis modules dynamically or directly
mod_01 = import_module("01_negative_flow")
mod_02 = import_module("02_spikes")
mod_03 = import_module("03_unrealistic_flow")
mod_04 = import_module("04_leakage")
mod_05 = import_module("05_zero_consumption")
mod_06 = import_module("06_step_change")
mod_07 = import_module("07_magnetic_events")
mod_08 = import_module("08_custom_insights")


def generate_markdown_report(output_filename="REPORT.md") -> str:
    print("Loading water telemetry dataset...")
    df_pok, df_prib, df_sob = load_full_dataset()
    print("Running analytical anomaly detection modules...")

    res_01 = mod_01.analyze_negative_flow(df_pok, df_prib)
    res_02 = mod_02.analyze_spikes(df_pok, df_prib)
    res_03 = mod_03.analyze_unrealistic_flow(df_pok, df_prib)
    res_04 = mod_04.analyze_leakage(df_pok, df_prib)
    res_05 = mod_05.analyze_zero_consumption(df_pok, df_prib)
    res_06 = mod_06.analyze_step_change(df_pok, df_prib)
    res_07 = mod_07.analyze_magnetic_events(df_pok, df_prib, df_sob)
    res_08 = mod_08.analyze_custom_insights(df_pok, df_prib, df_sob)

    md = []
    md.append("# Аналитический отчет: Поиск аномалий в потреблении воды")
    md.append("**Роль:** Senior Data Analyst  ")
    md.append("**Предмет исследования:** Анализ телеметрических данных приборов учета воды (3 000 приборов, 693 443 записи показаний за 1 год).  ")
    md.append("**Исходные данные:** `test/pokazaniya.csv`, `test/pribory.csv`, `test/sobytiya.csv`.  \n")
    md.append("---\n")

    md.append("## Обзор результатов аналитики\n")
    md.append("| # | Тип аномалии | Порог / Критерий | Найдено приборов | Ключевая причина |")
    md.append("|---|---|---|---|---|")
    md.append(f"| 1 | **Отрицательный расход** | `raw_diff < -0.001 м³` | **{res_01['total_affected_devices']}** | Битовые сбои телеметрии / сброс регистра / обратный ток |")
    md.append(f"| 2 | **Всплески потребления** | `3.0 < daily_flow <= 50.0 м³` И `>= 5x медианы` | **{res_02['total_affected_devices']}** | Залповый расход / авария арматуры / пропуск данных |")
    md.append(f"| 3 | **Нереальные значения** | `raw_diff > 50.0 м³/сут` | **{res_03['total_affected_devices']}** | Переполнение 32-битного счетчика (integer overflow) |")
    md.append(f"| 4 | **Утечка (без отдыха)** | `>= 90 дней` непрерывного расхода | **{res_04['total_affected_devices']}** | Неисправность бачка/крана / микротрещина трубы |")
    md.append(f"| 5 | **Длительный ноль** | `>= 90 дней` zero-flow при наличии пакетов | **{res_05['total_affected_devices']}** | Незаселенная квартира / заклинивание крыльчатки |")
    md.append(f"| 6 | **Ступенька (сдвиг)** | Изменение среднего в `>= 3x` раз (сдвиг `>= 0.15 м³`) | **{res_06['total_affected_devices']}** | Изменение состава жильцов / новая утечка |")
    md.append(f"| 7 | **Магнитные события** | Падение расхода на `>= 50%` после 'магнита' | **{res_07['total_affected_devices']}** | Несанкционированное вмешательство (хищение воды) |")
    md.append(f"| 8 | **Собственные находки** | Пакетные штормы, сбои ревизий, перезагрузки | **3 ключевых тренда** | Аппаратная уязвимость старой ревизии AQUA2 |")
    md.append("\n---\n")

    # SECTION 1
    md.append("### 1. Отрицательный расход (Meter Backwards)")
    md.append(f"**Метод и критерий:** {res_01['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** {res_01['total_affected_devices']} (всего {res_01['total_anomalous_records']} аномальных записей).  ")
    md.append("**Примеры аномальных приборов (anon_id):**")
    for ex in res_01["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Мин. перепад: **{ex['min_negative_diff']:,.3f} м³** | Модель: `{ex['tip']}` ({ex['model']}) | Первое проявление: `{ex['first_negative_date'].strftime('%Y-%m-%d')}`")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_01['causes_explanation']}\n")

    # SECTION 2
    md.append("### 2. Всплески потребления (Daily Flow Spikes)")
    md.append(f"**Метод и критерий:** {res_02['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** {res_02['total_affected_devices']} (всего {res_02['total_anomalous_records']} аномальных всплесков).  ")
    md.append("**Примеры аномальных приборов (anon_id):**")
    for ex in res_02["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Макс. суточный расход: **{ex['max_spike_flow']:.2f} м³** (Нормальная медиана: `{ex['median_daily_flow']:.3f} м³`, Превышение: **{ex['max_spike_ratio']:.1f}x**) | `{ex['tip']}`")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_02['causes_explanation']}\n")

    # SECTION 3
    md.append("### 3. Нереальные значения (Physically Impossible Values)")
    md.append(f"**Метод и критерий:** {res_03['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** {res_03['total_affected_devices']} (всего {res_03['total_anomalous_records']} нереальных измерений).  ")
    md.append("**Примеры аномальных приборов (anon_id):**")
    for ex in res_03["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Максимальный зафиксированный расход: **{ex['max_unrealistic_flow']:,.2f} м³/сут** | Модель: `{ex['tip']}` ({ex['model']})")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_03['causes_explanation']}\n")

    # SECTION 4
    md.append("### 4. Утечка (Continuous Non-Zero Flow)")
    md.append(f"**Метод и критерий:** {res_04['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** **{res_04['total_affected_devices']}** (при пороге >= 90 дней).  ")
    md.append(f"**Распределение по длительности утечки:** " + ", ".join([f"`{k}`: {v}" for k, v in res_04["threshold_counts"].items()]) + ".  ")
    md.append("**Примеры приборов с длительной утечкой (anon_id):**")
    for ex in res_04["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Дней подряд без перерыва: **{int(ex['max_positive_streak'])} дней** | Средний расход при утечке: `{ex['avg_leak_flow']:.3f} м³/сут` | `{ex['tip']}`")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_04['causes_explanation']}\n")

    # SECTION 5
    md.append("### 5. Длительное нулевое потребление при активной телеметрии")
    md.append(f"**Метод и критерий:** {res_05['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** **{res_05['total_affected_devices']}** (при пороге >= 90 дней).  ")
    md.append(f"**Распределение по длительности нулевого расхода:** " + ", ".join([f"`{k}`: {v}" for k, v in res_05["threshold_counts"].items()]) + ".  ")
    md.append("**Примеры приборов с длительным нолем (anon_id):**")
    for ex in res_05["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Дней нулевого расхода подряд: **{int(ex['max_zero_streak'])} дней** | Передано пакетов: `{int(ex['total_packets_sent'])}` | `{ex['tip']}`")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_05['causes_explanation']}\n")

    # SECTION 6
    md.append("### 6. Ступенька (Step Change / Sustained Baseline Shift)")
    md.append(f"**Метод и критерий:** {res_06['criterion']}  ")
    md.append(f"**Количество обнаруженных приборов:** **{res_06['total_affected_devices']}** (Резкий рост: {res_06['step_up_count']}, Резкое падение: {res_06['step_down_count']}).  ")
    md.append("**Примеры приборов со ступенькой (anon_id):**")
    for ex in res_06["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — `{ex['step_type']}`: До: `{ex['mean1']:.3f} м³/сут` -> После: `{ex['mean2']:.3f} м³/сут` (Кратность: **{ex['ratio']:.1f}x**) | `{ex['tip']}`")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_06['causes_explanation']}\n")

    # SECTION 7
    md.append("### 7. Сопоставление с событиями 'Магнит' (Magnetic Tampering)")
    md.append(f"**Метод и критерий:** {res_07['criterion']}  ")
    md.append(f"**Общая статистика магнитных срабатываний:** Всего **{res_07['total_mag_event_records']:,}** срабатываний на **{res_07['unique_devices_with_magnet']}** приборах.  ")
    md.append(f"**Количество подтвержденных приборов с падением расхода:** **{res_07['total_affected_devices']}** приборов (в т.ч. **{res_07['devices_dropped_to_zero_count']}** приборов упали до полного НУЛЯ).  ")
    md.append("**Примеры приборов с вероятным хищением воды (anon_id):**")
    for ex in res_07["top_examples"]:
        md.append(f"- `anon_id: {ex['anon_id']}` — Расход до магнита: `{ex['mean_before']:.3f} м³/сут` -> После: `{ex['mean_after']:.3f} м³/сут` (Падение на **{(1-ex['ratio'])*100:.1f}%**) | `{ex['total_mag_events']}` сработок магнита")
    md.append(f"\n**Аналитическое объяснение:**  \n{res_07['causes_explanation']}\n")

    # SECTION 8
    ins = res_08
    md.append("### 8. Собственные аналитические находки")
    
    # Insight 1
    i1 = ins["insight_1"]
    md.append(f"#### 8.1. {i1['title']}")
    md.append(f"**Критерий:** {i1['criterion']}  ")
    md.append(f"**Обнаружено:** **{i1['total_affected_devices']}** приборов осуществляют аномальную частоту выхода в эфир (до 2 585 пакетов/сут).  ")
    md.append(f"**Примеры anon_id:** " + ", ".join([f"`{ex['anon_id']}` ({ex['max_packets']} пак/сут)" for ex in i1["top_examples"]]) + ".  ")
    md.append(f"**Вывод:** {i1['explanation']}\n")

    # Insight 2
    i2 = ins["insight_2"]
    md.append(f"#### 8.2. {i2['title']}")
    md.append(f"**Вывод аналитика:** {i2['explanation']}  ")
    md.append("##### Сравнительная таблица надежности ревизий приборов:")
    md.append("| Тип прибора (tip) | Модель (model) | Всего приборов | Сбои выбросов (>50м³) | Отрицательный расход | % Сбойных приборов |")
    md.append("|---|---|---|---|---|---|")
    for _, r in i2["model_table"].iterrows():
        md.append(f"| `{r['tip']}` | `{r['model']}` | {r['total_devices']} | {r['unreal_devices']} | {r['neg_devices']} | **{r['anomaly_pct']:.2f}%** |")
    md.append("\n")

    # Insight 3
    i3 = ins["insight_3"]
    md.append(f"#### 8.3. {i3['title']}")
    md.append(f"**Критерий:** {i3['criterion']}  ")
    md.append(f"**Обнаружено:** **{i3['total_affected_devices']}** приборов перенесли **{i3['total_reset_events']}** аппаратных перезагрузок ('сброс').  ")
    md.append(f"**Примеры anon_id:** " + ", ".join([f"`{ex['anon_id']}` ({ex['reset_count']} сбросов)" for ex in i3["top_examples"]]) + ".  ")
    md.append(f"**Вывод:** {i3['explanation']}\n")

    md.append("---\n")
    md.append("## ПРИОРИТЕТНЫЙ ПЛАН ИНСПЕКЦИИ И ПРОВЕРКИ ПРИБОРОВ\n")
    md.append("Как Senior Data Analyst, я рекомендую сформировать наряды на выездную инспекцию в следующем порядке приоритетности:\n")
    
    md.append("### 🔴 ПРИОРИТЕТ 1: Подозрение на хищение воды и магнитное вмешательство")
    md.append(f"- **Кого проверять:** **27 приборов** с подтвержденным падением расхода после срабатывания магнитного датчика (в первую очередь 11 приборов с полным нулем: `3692442`, `8521337`, `5490175`, `6377542`, `8236638`).")
    md.append("- **Действие:** Выезд контролера для составления акта о несанкционированном вмешательстве в работу прибора учета, проверка целостности антимагнитных пломб и доначисление по нормативу.\n")

    md.append("### 🟠 ПРИОРИТЕТ 2: Аппаратные сбои и переполнение памяти (Ревизия AQUA2)")
    md.append(f"- **Кого проверять:** **70 приборов** с нереальными расходами (> 50 м³) и **102 прибора** с отрицательным расходом (в первую очередь `2519659`, `2491162`, `8908023`, `2159987`, `7891133`).")
    md.append("- **Действие:** Обновление прошивки контроллера либо плановая замена устаревших приборов модификации `AQUA2` на надежную ревизию `aqua2_nvt_wa_stm`.\n")

    md.append("### 🟡 ПРИОРИТЕТ 3: Длительные скрытые утечки воды")
    md.append(f"- **Кого проверять:** **939 приборов** с непрерывным расходом без отдыха >= 90 дней (в т.ч. 146 приборов без отдыха 365 дней: `1968874`, `7496751`, `2931056`).")
    md.append("- **Действие:** Уведомление абонента о наличии скрытой утечки через личный кабинет/SMS для предотвращения затопления и финансовых потерь.\n")

    md.append("### 🔵 ПРИОРИТЕТ 4: Застрявшие счетчики / Нежилой фонд")
    md.append(f"- **Кого проверять:** **524 прибора** с нулевым расходом >= 90 дней при активной телеметрии (в т.ч. `7558835`, `9686922`, `2072982`).")
    md.append("- **Действие:** Проверка физического вращения крыльчатки при открытом кране (выявление заклинивания механики солевыми отложениями).\n")

    report_content = "\n".join(md)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Report successfully saved to {output_filename}")
    return report_content


if __name__ == "__main__":
    generate_markdown_report()
