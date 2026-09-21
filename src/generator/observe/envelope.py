from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import EVENT_TYPE_SOURCE, PAYLOAD_FIELDS, PAYLOAD_REQUIRED
from ..rng import stable_hash


# ============================================================
# КОНВЕРТ СОБЫТИЯ
# ============================================================
#
#   event_id     устойчивый идентификатор записи; встречается
#                в выгрузке ровно один раз
#   client_id    чей это факт
#   event_type   что произошло
#   source       какая система банка записала факт
#   event_time   ТОЧНОЕ время, когда действие или факт
#                произошли; другого времени у записи нет
#   payload      остальное, JSON-строкой
#
# Шесть колонок и ничего больше. Запись сразу окончательна:
# исправлений, версий и повторных доставок не бывает, поэтому
# ни версии, ни метки доставки в конверте нет.
#
# Деловая связь живёт ключами payload: cause_event_id называет
# событие-причину, contract_id, application_id, case_id,
# offer_id, session_id и transfer_id — сущность, частью которой
# запись является. Отдельной метки связи конверт не несёт: вид
# связи задаёт имя ключа.
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
    event_id: str
    client_id: str
    event_type: str
    source: str
    event_time: datetime
    payload: dict


class EventFactory:
    """
    Выдаёт события одного клиента со стабильными
    идентификаторами.
    """

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self._counters: dict[str, int] = {}

    def next_id(self, source: str) -> str:
        index = self._counters.get(source, 0) + 1
        self._counters[source] = index
        return f"ev{stable_hash(self.client_id, source, index) % 10 ** 15:015d}"

    def make(self, event_type: str, ts: datetime, payload: dict) -> Event:

        source = EVENT_TYPE_SOURCE[event_type]

        allowed = PAYLOAD_FIELDS[event_type]

        unknown = sorted(set(payload) - allowed)

        if unknown:
            raise PayloadError(
                f"{event_type}: каталог не знает ключей {unknown}; "
                f"допустимы {sorted(allowed)}"
            )

        clean = {name: value for name, value in payload.items() if value is not None}

        empty = sorted(PAYLOAD_REQUIRED[event_type] - set(clean))

        if empty:
            raise PayloadError(f"{event_type}: обязательные ключи не заполнены: {empty}")

        return Event(
            event_id=self.next_id(source),
            client_id=self.client_id,
            event_type=event_type,
            source=source,
            event_time=ts,
            payload=clean,
        )


__all__ = ["Event", "EventFactory", "PayloadError"]
