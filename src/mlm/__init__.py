"""
Модель целиком и её обучение.

Начальные веса энкодеров события, анкеты и истории — без прохода по
данным:

    python -m src.mlm.init_backbone

Обучение на train — сквозной проход InputEmbedding -> Event Encoder ->
Profile Encoder -> History Encoder -> MLM, backward, клип и шаг AdamW
с warmup + cosine по всем весам модели:

    python -m src.mlm.train [--epochs N] [--max-steps N] [--resume]

Модель одна: входной слой из data/06_embeddings/train, начальные веса
из data/07_backbone, голова разыгрывается заново. Маска train
разыгрывается маскером src.masking заново на каждую эпоху. Чекпойнты —
data/12_train/checkpoint.pt (последний) и best_checkpoint.pt (лучший
val_loss).
"""
