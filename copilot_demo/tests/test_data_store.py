from backend import data_store as ds


def test_getters():
    ds.reload()
    changes = ds.get_changes()
    assert len(changes) == 15
    cid = changes[0]["change_id"]
    assert ds.get_change(cid)["change_id"] == cid
    assert len(ds.get_items()) == 200
    assert len(ds.get_stats(cid)) == 200
    assert len(ds.get_ranking(cid)) == 8
    assert len(ds.get_review_cards()) == 12


def test_review_card_lookup_returns_none_for_current():
    ds.reload()
    current = [c for c in ds.get_changes() if c["status"] == "current"][0]
    assert ds.get_review_card(current["change_id"]) is None
