"""
Живой дашборд обучения: отдельный процесс Streamlit, который читает
telemetry.jsonl каталога прогона, пока идёт обучение.

    python -m src.dashboard [--run data/runs/w4-b0] [--port 8501]

Обучение он не тормозит и в GPU не заходит: только дочитывает
новые строки файла. Чтение — src.dashboard.telemetry, страница —
src.dashboard.app.
"""
