"""data/*.json 로딩 + 인메모리 캐시. 파일 부재 시 자동 생성."""
import json

from backend.constants import DATA_DIR
from backend import mock_data

_CACHE = {}


def ensure_data():
    if not (DATA_DIR / "changes.json").exists():
        mock_data.generate_all()


def _load(name):
    if name not in _CACHE:
        ensure_data()
        _CACHE[name] = json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return _CACHE[name]


def reload():
    _CACHE.clear()


def get_changes():
    return _load("changes")


def get_change(cid):
    for c in get_changes():
        if c["change_id"] == cid:
            return c
    return None


def get_items():
    return _load("eds_items")


def get_stats(cid):
    return [s for s in _load("stats") if s["change_id"] == cid]


def get_ranking(cid):
    return [r for r in _load("ranking") if r["change_id"] == cid]


def get_review_cards():
    return _load("review_cards")


def get_review_card(cid):
    for card in get_review_cards():
        if card["change_id"] == cid:
            return card
    return None
