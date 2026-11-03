# F. Происхождение эталона (data/12_train/best_checkpoint.pt) — проверка controller'а, 2026-11-02

Чисто, расхождений нет:
- epoch 5, epoch_complete True, step 22475, best_val_loss 1.98117 (= телеметрия эталона).
- data-запись чекпойнта == текущему data_record(("train","val")): sha256 05_dataset/train и val, time_anchor обоих = cutoff, sha256 03_vocab/value_weights — все совпадают.
- config чекпойнта == MlmConfig() по умолчанию (пустой дифф); masking чекпойнта == MaskingConfig() по умолчанию.
- git log 0df3f63..HEAD -- src/ : 0 коммитов; рабочее дерево src/ меняется только новыми файлами LoRA (src/downstream/lora.py, finetune.py) — код модели и входа тот же, что у прогона.
- embed.trained_model сверяет только ключи train (src/downstream/embed.py:70-75) — подмножество, у эталона совпадает.
