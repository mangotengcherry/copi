from backend import ranking, data_store as ds


def test_ranking_sorted_desc_and_typed():
    ds.reload()
    cid = ds.get_changes()[0]["change_id"]
    table = ranking.get_ranking_table(cid)
    assert len(table) == 8
    scores = [r.risk_score for r in table]
    assert scores == sorted(scores, reverse=True)
    assert all(r.risk_level in ("High", "Med", "Low") for r in table)
