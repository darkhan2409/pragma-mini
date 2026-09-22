from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import EVENT_TYPE_SOURCE, PAYLOAD_FIELDS, PAYLOAD_REQUIRED


# ============================================================
# КОНВЕРТ СОБЫТИЯ
# ============================================================
#
#   client_id    чей это факт
#   event_time   ТОЧНОЕ время, когда действие или факт
#                произошли; другого времени у записи нет
#   source       какая система банка записала факт
#   payload      что произошло и всё остальное, JSON-строкой
#
# Четыре колонки и ничего больше. Тип события лежит ВНУТРИ
# payload под ключом type: отдельной колонки у него нет, и
# читатель узнаёт тип из самой записи.
#
# Идентификатора записи в конверте нет: тождество строки банку
# не нужно, а истории клиента хватает времени события. Запись
# сразу окончательна: исправлений, версий и повторных доставок
# не бывает.
#
# Деловая связь живёт деловыми ключами payload: contract_id,
# account_id, card_id, application_id, case_id, offer_id,
# session_id, transfer_id и merchant_id называют сущность,
# частью которой запись является. Ссылки на событие-причину
# нет: причинные связи не восстанавливаются ни полем, ни
# догадкой.
#
# Неприменимое поле ОТСУТСТВУЕТ. Не null, а именно отсутствует:
# ключ без значения не сообщает ничего, чего не сообщает его
# отсутствие, но занимает место в файле и превращается в токен
# «нет значения» дальше по конвейеру. Заполнять поле случайным
# значением ради плотности таблицы тем более запрещено.
# ============================================================


class PayloadError(Exception):
    """
    Payload не сходится с каталогом ключей.

    Это ошибка КОДА, а не данных: лишний ключ обычно означает
    опечатку в имени поля, пустой обязательный — забытую
    подстановку. Раньше make() молча выбрасывал лишнее и
    подставлял null вместо забытого, и обе ошибки доезжали до
    выгрузки незамеченными.
    """


@dataclass
class Event:
    client_id: str
    event_time: datetime
    source: str
    payload: dict
    # Тип события живёт в payload["type"]; здесь он повторён для
    # самой симуляции, которая спрашивает его на каждом шагу.
    # В выгрузку идёт только payload.
    event_type: str = ""
    # Номер выдачи внутри клиента. Живёт только в памяти
    # симуляции: держит порядок строк с одинаковым временем и
    # даёт воспроизводимый ключ случайности. В выгрузку не
    # попадает и заменой event_id не является.
    ordinal: int = 0


class EventFactory:
    """
    Выдаёт события одного клиента и нумерует их по порядку
    выдачи.
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self._ordinal = 0

    def make(self, event_type: str, ts: datetime, payload: dict) -> Event:

        source = EVENT_TYPE_SOURCE[event_type]

        allowed = PAYLOAD_FIELDS[event_type]

        unknown = sorted(set(payload) - allowed)

        if unknown:
            raise PayloadError(
                f"{event_type}: каталог не знает ключей {unknown}; "
                f"допустимы {sorted(allowed)}"
            )

        if "type" in payload:
            raise PayloadError(
                f"{event_type}: ключ type проставляет конверт, передавать его нельзя"
            )

        # Тип идёт первым ключом записи: читатель payload узнаёт,
        # что перед ним, до разбора остальных полей.
        clean = {"type": event_type}
        clean.update({name: value for name, value in payload.items() if value is not None})

        empty = sorted(PAYLOAD_REQUIRED[event_type] - set(clean))

        if empty:
            raise PayloadError(f"{event_type}: обязательные ключи не заполнены: {empty}")

        self._ordinal += 1

        return Event(
            client_id=self.client_id,
            event_time=ts,
            source=source,
            payload=clean,
            event_type=event_type,
            ordinal=self._ordinal,
        )


__all__ = ["Event", "EventFactory", "PayloadError"]
