from backend import similarity as sim, data_store as ds


def test_tree_proximity_levels():
    a = {"level1": "FE", "level2": "CVD", "level3": "GateOx"}
    assert sim.tree_proximity(a, a) == 1.0
    assert sim.tree_proximity(a, {"level1": "FE", "level2": "CVD", "level3": "STI"}) == 0.6
    assert sim.tree_proximity(a, {"level1": "FE", "level2": "Etch", "level3": "STI"}) == 0.3
    assert sim.tree_proximity(a, {"level1": "BE", "level2": "Metal", "level3": "M1"}) == 0.0


def test_signature_similarity_direction_bonus():
    base = {"WL": "up", "Vth": "down"}
    same = sim.signature_similarity(base, {"WL": "up", "Vth": "down"})
    opp = sim.signature_similarity(base, {"WL": "down", "Vth": "up"})
    assert same > opp
    assert sim.signature_similarity({}, {}) == 0.0


def test_find_similar_returns_top3_from_past():
    ds.reload()
    current = [c for c in ds.get_changes() if c["status"] == "current"][0]
    resp = sim.find_similar(current["change_id"])
    assert len(resp.candidates) <= 3
    ids = {c.change_id for c in resp.candidates}
    past_ids = {c["change_id"] for c in ds.get_changes() if c["status"] == "past"}
    assert ids <= past_ids
    # breakdown 축 노출 + 정렬
    scores = [c.total_score for c in resp.candidates]
    assert scores == sorted(scores, reverse=True)
    if resp.candidates:
        b = resp.candidates[0].breakdown
        assert 0.0 <= b.tree <= 1.0
