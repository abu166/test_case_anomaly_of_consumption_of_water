# Анализ аномалий потребления воды

## Требования
- Python 3.8+
- Зависимости из `requirements.txt` (`pandas`, `numpy`, `scipy`)

## Запуск проекта

```bash
# 1. Создание виртуального окружения и установка зависимостей
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Запуск анализа и генерация отчета
python main.py
```

Результат анализа сохраняется в файл `REPORT.md`.

## Структура модулей
- `data_loader.py` — загрузка данных из `test/` и расчет суточного расхода
- `01_negative_flow.py` — 1. Отрицательный расход
- `02_spikes.py` — 2. Всплески потребления
- `03_unrealistic_flow.py` — 3. Нереальные значения (>50 м³/сут)
- `04_leakage.py` — 4. Непрерывные утечки
- `05_zero_consumption.py` — 5. Длительное нулевое потребление
- `06_step_change.py` — 6. Ступенчатое изменение расхода
- `07_magnetic_events.py` — 7. Сопоставление со срабатыванием магнита
- `08_custom_insights.py` — 8. Собственные аналитические находки
- `generate_report.py` — формирование Markdown-отчета
- `main.py` — главный скрипт запуска
