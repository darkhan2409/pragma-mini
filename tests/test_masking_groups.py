from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest

from src.masking.apply import IGNORE, apply_selection
from src.masking.choose import EVENT, KEY, REASONS, VALUE, FieldGroups, choose, field_groups
from src.masking.settings import ConfigError, MaskingConfig
from src.masking.weights import load_value_weights
from src.mlm.inputs import Source
from src.tokenization.schema import SemanticSchema

from tests import world
from tests.test_masking import row_of
from tests.test_training_math import settle


# ============================================================
# ИДЕЯ
# ============================================================
#
# Зависимые поля одного события подсказывают друг друга: по
# merchant_name видно mcc, по direction — знак суммы. Если цель
# стала целью механизмом value или key, а её партнёр по группе в
# том же событии виден, модель списывает ответ, а не
# восстанавливает его. Группа закрывается целиком: все
# присутствующие поля группы в этом событии становятся целями.
#
# event_type при флаге — цель только механизма event: его значение
# видно в других полях события (тип операции, канал), а сам он
# подсказывает их. Флаг — фильтр после всех розыгрышей, поэтому
# остальные цели и их [UNK] не сдвигаются ни на бит.
#
# Мир здесь — синтетические клиенты на ключах key_a, key_b, key_c;
# для event_type берётся свой словарь, где этим именем назван
# key_c.
# ============================================================


VOCAB = {"key_a": world.KEY_A, "key_b": world.KEY_B, "key_c": world.KEY_C}

# Тот же мир, но key_c играет event_type.
TYPED = {"key_a": world.KEY_A, "key_b": world.KEY_B, "event_type": world.KEY_C}

GROUP = (("key_a", "key_b"),)

# Ключ анкеты: в словаре он есть, но ключом события не является.
PROFILE_KEY = "profile_city"


def schema_of(*events: str) -> SemanticSchema:
    """
    Реестр смыслов мира: ключи событий events и ключ анкеты. Происхождение
    задано так же, как в настоящем реестре: payload — поле события,
    profile.* — поле анкеты.
    """

    keys = {name: {"value_kind": "categorical", "physical_fields": [f"world.{name}"]} for name in events}
    keys[PROFILE_KEY] = {"value_kind": "categorical", "physical_fields": ["profile.city"]}

    return SemanticSchema({"keys": keys})


SCHEMA = schema_of("key_a", "key_b", "key_c")

TYPED_SCHEMA = schema_of("key_a", "key_b", "event_type")

SEEDS = range(12)


def mixed(**overrides) -> MaskingConfig:
    """
    Все механизмы сразу и заметный [UNK]: правило группы обязано
    держаться на любом их сочетании.
    """

    base = dict(
        seed=5,
        value_probability=0.3,
        event_probability=0.1,
        key_probability=0.25,
        unknown_probability=0.2,
        key_context_corruption_probability=0.5,
        informativeness_weighted_masking=False,
    )

    base.update(overrides)

    return MaskingConfig(**base)


def people(seed: int, count: int = 30, prefix: str = "g") -> list[world.Made]:
    """
    Клиенты случайной формы: в событии — случайное подмножество
    ключей, значения из одного–трёх кусков, часть событий вне
    периода целей.
    """

    rng = np.random.default_rng(seed)

    made = []

    for number in range(count):

        events = []

        for _ in range(int(rng.integers(1, 7))):
            events.append([
                (key, [int(piece) for piece in rng.integers(10, 60, size=int(rng.integers(1, 4)))], False)
                for key in (world.KEY_A, world.KEY_B, world.KEY_C)
                if rng.random() < 0.7
            ])

        allowed = tuple(bool(rng.random() < 0.75) for _ in events)

        made.append(world.make(f"{prefix}{seed}-{number}", events, [(world.KEY_A, [20])], targetable=allowed))

    return made


def masked(row: dict, config: MaskingConfig, fields: FieldGroups | None, group: str = "train") -> dict:
    selection = choose(group, row, config, None, fields)
    return apply_selection(row["client_id"], row, selection, world.MASK, world.UNK)


def spans(key_ids, positions, starts, lengths) -> list[tuple[int, int, int, int]]:
    """
    Значения ленты: (событие, ключ, начало, конец), маркер не входит.
    """

    found = []

    for event, (start, length) in enumerate(zip(list(starts), list(lengths))):

        opened = -1

        for index in range(start + 1, start + length):

            if positions[index] != 0:
                continue

            if opened >= 0:
                found.append((event, int(key_ids[opened]), opened, index))

            opened = index

        if opened >= 0:
            found.append((event, int(key_ids[opened]), opened, start + length))

    return found


def visible_partners(key_ids, positions, starts, lengths, value_ids, labels, members: set[int],
                     unknown_allowed: bool) -> list:
    """
    Нарушения правила группы: цель ключа группы, рядом с которой в том
    же событии поле той же группы не цель. При unknown_allowed
    партнёр в [UNK] не нарушение: он не виден.
    """

    found = spans(key_ids, positions, starts, lengths)

    broken = []

    for event, key, start, end in found:

        if key not in members or labels[start] == IGNORE:
            continue

        for other_event, other_key, other_start, other_end in found:

            if other_event != event or other_key not in members or other_start == start:
                continue

            closed = all(labels[index] != IGNORE and value_ids[index] == world.MASK
                         for index in range(other_start, other_end))

            hidden = unknown_allowed and all(value_ids[index] == world.UNK
                                             for index in range(other_start, other_end))

            if not (closed or hidden):
                broken.append((event, key, other_key))

    return broken


# ============================================================
# ПРАВИЛО ГРУППЫ
# ============================================================


@pytest.mark.parametrize("unknown", [0.0, 0.2])
def test_no_target_has_a_visible_group_partner(stage, monkeypatch, unknown: float):
    """
    У каждой цели value/key все поля её группы в том же событии —
    цели: метка есть, на входе [MASK]. Партнёр, ушедший в [UNK] ещё
    до замыкания, тоже не виден — при unknown 0 таких нет вовсе.
    Проверка — и маскером напрямую, и через Source, в том числе
    после переезда Source в процесс подготовки (pickle).
    """

    config = mixed(field_groups=GROUP, unknown_probability=unknown)
    fields = field_groups(config, VOCAB, SCHEMA)

    members = {world.KEY_A, world.KEY_B}

    closed = 0
    exposed = 0

    for seed in SEEDS:
        for made in people(seed):

            row = row_of(made)
            out = masked(row, config, fields)
            base = masked(row, config, None)

            client = made.client
            shape = (client.key_ids, client.positions, client.event_starts, client.event_lengths)

            assert visible_partners(*shape, out["value_ids"], out["labels"], members, unknown > 0) == []

            closed += int((out["labels"] != IGNORE).sum() - (base["labels"] != IGNORE).sum())

            # Без групп у той же маски видимые партнёры есть — иначе
            # проверка выше ничего не доказывала бы.
            exposed += len(visible_partners(*shape, base["value_ids"], base["labels"], members, unknown > 0))

    # Правило обязано было сработать: целей с группой больше.
    assert closed > 0
    assert exposed > 0

    settle(stage, train_people=people(3, count=40, prefix="s"))

    # Ключей мира нет в настоящем реестре смыслов: Source берёт реестр мира.
    monkeypatch.setattr(SemanticSchema, "open", staticmethod(lambda group=None: SCHEMA))

    source = Source("train", masking=config)
    moved = pickle.loads(pickle.dumps(source))

    for reader in (source, moved):

        clients = list(reader.clients())

        assert any(client.n_targets for client in clients)

        for client in clients:
            assert visible_partners(client.key_ids, client.positions, client.event_starts, client.event_lengths,
                                    client.value_ids, client.labels, members, unknown > 0) == []

    assert [client.labels.tolist() for client in source.clients()] == [
        client.labels.tolist() for client in moved.clients()
    ]


def test_group_member_absent_or_multi_piece():
    """
    Событие без части полей группы закрывает только присутствующие, и
    ни одной позиции больше. Текст из нескольких кусков закрывается
    всеми кусками, метка — на каждом. Ожидаемый выбор считается здесь
    обходом: выбор без групп плюс партнёры источников value/key не в
    [UNK], с сильнейшей причиной источника.
    """

    made = world.make(
        "absent",
        [
            # Вся группа, key_b — текст из трёх кусков.
            [(world.KEY_A, [10], False), (world.KEY_B, [11, 12, 13], False), (world.KEY_C, [30], False)],
            # key_b нет.
            [(world.KEY_A, [14], False), (world.KEY_C, [31], False)],
            # Только key_b, снова из трёх кусков.
            [(world.KEY_B, [15, 16, 17], False)],
            # Только key_c — вне группы.
            [(world.KEY_C, [32, 33], False)],
        ],
        [(world.KEY_A, [20])],
    )

    row = row_of(made)
    client = made.client

    found = spans(client.key_ids, client.positions, client.event_starts, client.event_lengths)
    of_key = {world.KEY_A: 0, world.KEY_B: 0}

    multi_closed = 0
    lone = 0

    for seed in range(400):

        config = mixed(seed=seed, field_groups=GROUP, value_probability=0.25, key_probability=0.15)
        fields = field_groups(config, VOCAB, SCHEMA)

        base = choose("train", row, config)
        grouped = choose("train", row, config, None, fields)

        expected = {int(index): (int(code), bool(unknown))
                    for index, code, unknown in zip(base.picked, base.reasons, base.unknown)}

        strongest: dict[int, int] = {}

        for index, (code, unknown) in expected.items():
            event, key = found[index][0], found[index][1]
            if REASONS[code] in (KEY, VALUE) and not unknown and key in of_key:
                strongest[event] = min(strongest.get(event, code), code)

        for index, (event, key, _, _) in enumerate(found):
            if key in of_key and event in strongest and index not in expected:
                expected[index] = (strongest[event], False)

        assert grouped.picked.tolist() == sorted(expected)
        assert list(zip(grouped.reasons.tolist(), grouped.unknown.tolist())) == [
            expected[index] for index in sorted(expected)
        ]

        out = apply_selection(row["client_id"], row, grouped, world.MASK, world.UNK)

        for index, (event, key, start, end) in enumerate(found):

            pieces = range(start, end)

            if index in expected and not expected[index][1]:
                assert all(out["value_ids"][place] == world.MASK for place in pieces)
                assert [out["labels"][place] for place in pieces] == row["value_ids"][start:end]
            elif index in expected:
                assert all(out["value_ids"][place] == world.UNK for place in pieces)
                assert all(out["labels"][place] == IGNORE for place in pieces)
            else:
                assert [out["value_ids"][place] for place in pieces] == row["value_ids"][start:end]
                assert all(out["labels"][place] == IGNORE for place in pieces)

        added = set(expected) - set(base.picked.tolist())

        # key_b из трёх кусков закрыт партнёром key_a.
        multi_closed += int(any(found[index][1] == world.KEY_B and found[index][0] == 0 for index in added))

        # Событие без key_b: источник key_a, закрывать нечего.
        lone += int(1 in strongest)

    assert multi_closed > 0
    assert lone > 0


def test_partner_takes_the_strongest_source_reason():
    """
    Группа из трёх ключей; в событии один член выбран механизмом key,
    другой — value, третий свободен. Свободный закрывается с причиной
    key — сильнейшей из источников (меньший ненулевой код REASONS), а
    источники свои причины сохраняют. Проверяется при любом порядке
    членов в событии.
    """

    made = world.make(
        "three",
        [[(world.KEY_A, [10], False), (world.KEY_B, [11, 12], False), (world.KEY_C, [13], False)]],
        [(world.KEY_A, [20])],
    )

    row = row_of(made)
    trio = (("key_a", "key_b", "key_c"),)

    cases: set[int] = set()

    for seed in range(3000):

        config = mixed(seed=seed, field_groups=trio, event_probability=0.0, unknown_probability=0.0,
                       key_probability=0.3, value_probability=0.3)

        base = choose("train", row, config)

        reasons = {int(index): REASONS[code] for index, code in zip(base.picked, base.reasons)}

        if sorted(reasons.values()) != [KEY, VALUE]:
            continue

        free = ({0, 1, 2} - set(reasons)).pop()

        grouped = choose("train", row, config, None, field_groups(config, VOCAB, SCHEMA))

        assert grouped.picked.tolist() == [0, 1, 2]
        assert grouped.unknown.tolist() == [False, False, False]

        closed = {int(index): REASONS[code] for index, code in zip(grouped.picked, grouped.reasons)}

        assert closed == {**reasons, free: KEY}

        cases.add(free)

    # Свободным побывал каждый член группы.
    assert cases == {0, 1, 2}


# ============================================================
# EVENT_TYPE ТОЛЬКО ПРИ EVENT
# ============================================================


def test_event_type_is_a_target_only_by_event():
    """
    При флаге у event_type нет целей с причиной key или value, цели с
    причиной event есть. Контекст event_type вне целей не портится:
    его значения там остаются исходными. Без флага та же маска даёт
    event_type и цели key/value, и испорченный контекст.
    """

    on = mixed(event_type_event_only=True)
    off = mixed()

    fields = field_groups(on, TYPED, TYPED_SCHEMA)

    assert field_groups(off, TYPED, TYPED_SCHEMA) is None

    reasons_on: set[str] = set()
    reasons_off: set[str] = set()
    spoiled_on = 0
    spoiled_off = 0

    for seed in SEEDS:
        for made in people(seed):

            row = row_of(made)
            client = made.client
            allowed = world.targets_of(made)

            for config, used, reasons in ((on, fields, reasons_on), (off, None, reasons_off)):

                out = masked(row, config, used)

                for event, key, start, end in spans(client.key_ids, client.positions,
                                                    client.event_starts, client.event_lengths):

                    if key != world.KEY_C:
                        continue

                    if out["labels"][start] != IGNORE:
                        reasons.add(out["reason"][start])

                    if not allowed[event] and out["value_ids"][start] == world.UNK:
                        if used is None:
                            spoiled_off += 1
                        else:
                            spoiled_on += 1

    assert reasons_on == {EVENT}
    assert {KEY, VALUE} <= reasons_off
    assert spoiled_on == 0
    assert spoiled_off > 0


def test_event_type_event_only_leaves_other_values_alone():
    """
    Без групп, флаг вкл и выкл: у всех значений, кроме event_type, —
    те же цели, причины, [UNK] и испорченный контекст, побитно. У
    event_type совпадает всё, что выбрано механизмом event.
    """

    on = mixed(event_type_event_only=True)
    off = mixed()

    fields = field_groups(on, TYPED, TYPED_SCHEMA)

    differs = 0

    for seed in SEEDS:
        for made in people(seed):

            row = row_of(made)
            client = made.client

            first = masked(row, on, fields)
            second = masked(row, off, None)

            for event, key, start, end in spans(client.key_ids, client.positions,
                                                client.event_starts, client.event_lengths):

                pieces = slice(start, end)

                if key == world.KEY_C and second["reason"][start] != EVENT:
                    differs += int(first["value_ids"][pieces].tolist() != second["value_ids"][pieces].tolist())
                    continue

                assert first["value_ids"][pieces].tolist() == second["value_ids"][pieces].tolist()
                assert first["labels"][pieces].tolist() == second["labels"][pieces].tolist()
                assert first["reason"][pieces] == second["reason"][pieces]

    # Флаг что-то менял — у event_type.
    assert differs > 0


# ============================================================
# ПРЕЖНЯЯ МАСКА
# ============================================================


def test_groups_off_is_the_previous_mask():
    """
    field_groups=() и флаг выключен: FieldGroups не строится, а choose с
    fields=None и без аргумента дают побитно один выбор — и со
    взвешиванием, и без. Пустые группы, поданные явно, тоже ничего не
    меняют.
    """

    weights = load_value_weights()

    for config in (mixed(), mixed(informativeness_weighted_masking=True)):

        assert field_groups(config, VOCAB, SCHEMA) is None

        empty = FieldGroups(of_key=np.full(world.KEY_C + 1, -1, dtype=np.int64), event_type=-1)

        used = weights if config.informativeness_weighted_masking else None

        for seed in SEEDS:
            for made in people(seed):

                row = row_of(made)

                plain = choose("train", row, config, used)

                for other in (choose("train", row, config, used, None), choose("train", row, config, used, empty)):
                    assert other.picked.tolist() == plain.picked.tolist()
                    assert other.reasons.tolist() == plain.reasons.tolist()
                    assert other.unknown.tolist() == plain.unknown.tolist()
                    assert other.spoiled.start.tolist() == plain.spoiled.start.tolist()


# ============================================================
# КОНФИГ
# ============================================================


def test_field_groups_config_errors():
    """
    Неизвестный ключ, ключ в двух группах, группа из одного ключа —
    ConfigError. event_type при флаге обязан быть в словаре и не может
    стоять в группе: замыкание сделало бы его целью value/key.
    as_dict/from_dict — туда и обратно через JSON; словарь без новых
    ключей даёт значения по умолчанию.
    """

    with pytest.raises(ConfigError, match="nope"):
        field_groups(MaskingConfig(field_groups=(("key_a", "nope"),)), VOCAB, SCHEMA)

    # Ключ анкеты есть и в словаре, и в реестре, но ключом события не
    # является: в группе он молча ничего бы не закрывал.
    with pytest.raises(ConfigError, match=PROFILE_KEY):
        field_groups(MaskingConfig(field_groups=(("key_a", PROFILE_KEY),)), {**VOCAB, PROFILE_KEY: 8}, SCHEMA)

    # Ключ события по словарю, которого нет в реестре ключей событий.
    with pytest.raises(ConfigError, match="key_c"):
        field_groups(MaskingConfig(field_groups=(("key_a", "key_c"),)), VOCAB, TYPED_SCHEMA)

    with pytest.raises(ConfigError, match="key_b"):
        MaskingConfig(field_groups=(("key_a", "key_b"), ("key_b", "key_c"))).validate()

    with pytest.raises(ConfigError, match="key_a"):
        MaskingConfig(field_groups=(("key_a",),)).validate()

    with pytest.raises(ConfigError, match="event_type"):
        field_groups(MaskingConfig(event_type_event_only=True), VOCAB, SCHEMA)

    with pytest.raises(ConfigError, match="event_type"):
        MaskingConfig(field_groups=(("event_type", "key_a"),), event_type_event_only=True).validate()

    with pytest.raises(ConfigError):
        MaskingConfig.from_dict({"field_groups": [["key_a", "key_b"], ["key_c"]]})

    with pytest.raises(ConfigError):
        MaskingConfig.from_dict({"field_groups": ["key_a"]})

    config = MaskingConfig(field_groups=(("key_a", "key_b"), ("key_c", "key_d")), event_type_event_only=True)

    stored = json.loads(json.dumps(config.as_dict()))

    assert stored["field_groups"] == [["key_a", "key_b"], ["key_c", "key_d"]]
    assert stored["event_type_event_only"] is True
    assert MaskingConfig.from_dict(stored) == config

    old = {name: value for name, value in MaskingConfig().as_dict().items()
           if name not in ("field_groups", "event_type_event_only")}

    restored = MaskingConfig.from_dict(old)

    assert restored.field_groups == ()
    assert restored.event_type_event_only is False
    assert restored == MaskingConfig()

    fields = field_groups(MaskingConfig(field_groups=(("key_b", "key_c"),)), VOCAB, SCHEMA)

    assert fields.of_key[world.KEY_A] == -1
    assert fields.of_key[world.KEY_B] == fields.of_key[world.KEY_C] == 0
    assert fields.event_type == -1


# ============================================================
# ОТЧЁТ: ЦЕЛИ С ВИДИМЫМ ПАРТНЁРОМ
# ============================================================
#
# Мир отчёта: key_a — случайное поле, key_b — его копия (грубая
# функция от key_a: три значения на шесть), key_c — независимое.
# Тогда key_b определяется key_a, а обратное неверно.
# ============================================================


def copy_world(seed: int = 0, count: int = 40, prefix: str = "r") -> list[world.Made]:

    rng = np.random.default_rng(seed)

    made = []

    for number in range(count):

        events = []

        for _ in range(8):
            x = int(rng.integers(8, 14))
            events.append([
                (world.KEY_A, [x], False),
                (world.KEY_B, [26 + (x - 8) // 2], False),
                (world.KEY_C, [int(rng.integers(44, 50))], False),
            ])

        made.append(world.make(f"{prefix}{seed}-{number}", events, [(world.KEY_A, [20])]))

    return made


def test_report_finds_a_copied_field(stage, monkeypatch):
    """
    В мире с ключом-копией пара (key_b, key_a) найдена с отношением
    H(B|A)/H(B) ниже порога; обратной пары и пар с независимым
    key_c нет. Пары берутся по train, каким бы ни был запрошенный
    отчёт.
    """

    from src.masking.report import dependent_pairs, masking_report

    settle(stage, train_people=copy_world(), val_people=copy_world(1, prefix="v"))

    pairs = dependent_pairs()

    assert [(b, a) for b, a, _ in pairs] == [("key_b", "key_a")]
    assert pairs[0][2] < 0.05

    monkeypatch.setattr(SemanticSchema, "open", staticmethod(lambda group=None: SCHEMA))

    for group in ("train", "val"):
        assert masking_report(group, mixed())["partners"]["pairs"] == pairs

    # Порог выше — находится больше, чем при 0,05.
    assert len(dependent_pairs(threshold=0.9)) > len(pairs)


def test_report_partner_share_drops_to_zero_with_the_group(stage, monkeypatch, capsys):
    """
    Без групп у заметной доли целей в том же событии виден партнёр, и
    пара не покрыта. С группой (key_a, key_b) партнёр у цели не виден
    никогда (замыкание его прячет), и пара покрыта. Считаются только
    цели, а [UNK] и [MASK] не видны.
    """

    from src.masking.report import masking_report, show

    settle(stage, train_people=copy_world())

    monkeypatch.setattr(SemanticSchema, "open", staticmethod(lambda group=None: SCHEMA))

    report = masking_report("train", mixed())
    plain = report["partners"]
    grouped = masking_report("train", mixed(field_groups=GROUP))

    # Непокрытая пара видна в тексте, покрытая — нет.
    show(report)
    assert "key_b ← key_a" in capsys.readouterr().out

    show(grouped)
    assert "key_b ← key_a" not in capsys.readouterr().out

    assert plain["covered"] == [False]
    assert plain["targets_with_visible_partner"] > 0

    # Независимый пересчёт, как в check9 аудита: берутся примеры Source с
    # той же маской, и для цели ключа B пары (B, A) смотрится, виден ли в её
    # событии A (не [MASK] и не [UNK]).
    ids = {"key_a": world.KEY_A, "key_b": world.KEY_B, "key_c": world.KEY_C}
    wanted: dict[int, set[int]] = {}

    for b, a, _ in plain["pairs"]:
        wanted.setdefault(ids[b], set()).add(ids[a])

    for config, expected in ((mixed(), plain), (mixed(field_groups=GROUP), grouped["partners"])):

        source = Source("train", masking=config)
        counted = 0
        targets = 0

        for client in source.clients():
            for start in np.flatnonzero((client.labels != IGNORE) & (client.positions == 0)):

                targets += 1
                event = int(np.searchsorted(client.event_starts, start, side="right")) - 1
                first = int(client.event_starts[event])
                last = first + int(client.event_lengths[event])

                visible = {
                    int(client.key_ids[place])
                    for place in range(first + 1, last)
                    if client.positions[place] == 0 and client.value_ids[place] not in (source.mask_id, source.unknown_id)
                }

                counted += int(bool(wanted.get(int(client.key_ids[start]), set()) & visible))

        assert targets > 0
        assert counted == expected["targets_with_visible_partner"]
    assert 0.0 < plain["share"] <= 1.0
    assert plain["share"] == plain["targets_with_visible_partner"] / report["targets"]

    assert grouped["partners"]["covered"] == [True]
    assert grouped["partners"]["targets_with_visible_partner"] == 0
    assert grouped["partners"]["share"] == 0.0
    assert grouped["targets"] > 0


def test_cli_rejects_a_non_positive_pairs_row_groups(capsys):

    from src.masking.report import build_parser

    for bad in ("0", "-3", "много"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train", "--pairs-row-groups", bad])

        assert "не меньше 1" in capsys.readouterr().err

    assert build_parser().parse_args(["train", "--pairs-row-groups", "3"]).pairs_row_groups == 3


# ============================================================
# ПАРЫ: ГРАНИЧНЫЕ СЛУЧАИ
# ============================================================
#
# Мир собирается из событий напрямую: список полей на событие, по
# десять событий на клиента. Токены значений не проверяются словарём,
# поэтому значение может быть любой длины.
# ============================================================


def pair_world(events: list[list[world.Field]], per_client: int = 10) -> list[world.Made]:
    return [
        world.make(f"p{number}", events[number:number + per_client], [(world.KEY_A, [20])])
        for number in range(0, len(events), per_client)
    ]


def found_pairs(stage, events: list[list[world.Field]], **options) -> list[tuple[str, str, float]]:

    from src.masking.report import dependent_pairs

    settle(stage, train_people=pair_world(events))

    return dependent_pairs(**options)


def test_pairs_treat_a_multi_token_value_as_one_value(stage):
    """
    B = XOR двух кусков A: по любому одному куску B не определён, по
    кортежу из двух — определён. Пара найдена только если значение —
    кортеж токенов целиком.
    """

    rng = np.random.default_rng(0)

    events = []

    for _ in range(400):
        p, q = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        events.append([(world.KEY_A, [8 + p, 10 + q], False), (world.KEY_B, [26 + (p ^ q)], False)])

    pairs = found_pairs(stage, events)

    assert [(b, a) for b, a, _ in pairs] == [("key_b", "key_a")]
    assert pairs[0][2] == pytest.approx(0.0, abs=1e-12)


def test_pairs_use_only_events_with_both_keys(stage):
    """
    В половине событий A нет, а B случайно. Пара считается по событиям, где
    есть оба ключа, и в них B определён A. Если бы отсутствие A считалось
    ещё одним его значением, отношение было бы около 0,5.
    """

    rng = np.random.default_rng(1)

    events = []

    for _ in range(200):
        x = int(rng.integers(8, 14))
        events.append([(world.KEY_A, [x], False), (world.KEY_B, [26 + (x - 8) // 2], False)])
        events.append([(world.KEY_B, [26 + int(rng.integers(0, 3))], False)])

    pairs = found_pairs(stage, events)

    assert [(b, a) for b, a, _ in pairs] == [("key_b", "key_a")]
    assert pairs[0][2] == pytest.approx(0.0, abs=1e-12)


def test_pairs_ignore_a_near_unique_conditioning_key(stage, monkeypatch):
    """
    A почти уникален: меньше 20 событий на значение. По частотам H(B|A)
    у него ноль при любом B, и без запаса событий пара нашлась бы ложно;
    с запасом её нет. Что дело в запасе, видно по запасу в одно событие.
    """

    from src.masking import report

    rng = np.random.default_rng(2)

    events = [
        [(world.KEY_A, [int(piece) for piece in rng.integers(10, 60, size=3)], False),
         (world.KEY_B, [26 + int(rng.integers(0, 3))], False)]
        for _ in range(300)
    ]

    settle(stage, train_people=pair_world(events))

    assert report.dependent_pairs() == []

    monkeypatch.setattr(report, "EVENTS_PER_VALUE", 1)

    assert ("key_b", "key_a") in [(b, a) for b, a, _ in report.dependent_pairs()]


def test_pairs_ratio_matches_the_entropy_formula(stage):
    """
    Шумная пара, посчитанная вручную: A из трёх значений по 60 событий.
    a1 даёт b1, a2 даёт b2, a3 даёт b1 и b2 поровну. Тогда
    H(B|A) = (1/3) ln 2 при H(B) = ln 2, то есть 1/3; а H(A|B) — это
    H(2/3, 1/3) при H(A) = ln 3. Закрепляется и направление: (B, A) и (A, B)
    разные числа.
    """

    events = (
        [[(world.KEY_A, [8], False), (world.KEY_B, [26], False)]] * 60
        + [[(world.KEY_A, [9], False), (world.KEY_B, [27], False)]] * 60
        + [[(world.KEY_A, [10], False), (world.KEY_B, [26 + number % 2], False)] for number in range(60)]
    )

    by_pair = {(b, a): ratio for b, a, ratio in found_pairs(stage, events, threshold=1.0)}

    assert set(by_pair) == {("key_b", "key_a"), ("key_a", "key_b")}

    assert by_pair[("key_b", "key_a")] == pytest.approx(1 / 3, abs=1e-12)

    conditional = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))

    assert by_pair[("key_a", "key_b")] == pytest.approx(conditional / math.log(3), abs=1e-12)

    # Порог отсекает по отношению: 1/3 ниже 0,34 и выше 0,33.
    from src.masking.report import dependent_pairs

    assert [(b, a) for b, a, _ in dependent_pairs(threshold=0.34)] == [("key_b", "key_a")]
    assert dependent_pairs(threshold=0.33) == []
