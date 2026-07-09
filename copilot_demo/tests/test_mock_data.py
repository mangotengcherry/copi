import json
from backend import mock_data
from backend.constants import DATA_DIR, ITEM_GROUPS


def _load(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def test_generate_is_deterministic_and_counts():
    mock_data.generate_all()
    changes = _load("changes.json")
    items = _load("eds_items.json")
    stats = _load("stats.json")
    ranking = _load("ranking.json")
    cards = _load("review_cards.json")

    assert len(changes) == 15
    assert sum(c["status"] == "past" for c in changes) == 12
    assert sum(c["status"] == "current" for c in changes) == 3
    assert len(items) == 200
    assert {i["item_group"] for i in items} <= set(ITEM_GROUPS)
    assert len(stats) == 15 * 200
    assert len(ranking) == 15 * 8
    assert len(cards) == 12

    # 결정론성: 재생성해도 동일
    first = (DATA_DIR / "stats.json").read_text(encoding="utf-8")
    mock_data.generate_all()
    assert (DATA_DIR / "stats.json").read_text(encoding="utf-8") == first


def test_signal_structure_present():
    mock_data.generate_all()
    stats = _load("stats.json")
    # 각 change에 유의(q<0.1) item이 존재하고, 2~3개 group에 집중
    by_change = {}
    for s in stats:
        by_change.setdefault(s["change_id"], []).append(s)
    for cid, rows in by_change.items():
        sig = [r for r in rows if r["q_value"] < 0.1]
        assert len(sig) > 0
        sig_groups = {r["item_group"] for r in sig}
        assert 1 <= len(sig_groups) <= 4


def test_bh_monotonic():
    q = mock_data.benjamini_hochberg([0.001, 0.02, 0.5, 0.9])
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[0] <= q[-1]
