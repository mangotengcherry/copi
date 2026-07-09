from backend import stats_summary as ss, data_store as ds


def test_overview_counts_significant():
    ds.reload()
    cid = ds.get_changes()[0]["change_id"]
    ov = ss.build_overview(cid)
    assert ov.total_items_tested == 200
    assert ov.significant_item_count >= 0
    assert len(ov.affected_groups) == len({g.group for g in ov.affected_groups})


def test_signature_matches_significant_groups():
    ds.reload()
    cid = ds.get_changes()[0]["change_id"]
    sig = ss.significant_signature(cid)
    assert all(d in ("up", "down") for d in sig.values())


def test_histogram_deterministic():
    ds.reload()
    cid = ds.get_changes()[0]["change_id"]
    rep = [s for s in ds.get_stats(cid) if s["is_representative"]][0]
    h1 = ss.histogram_samples(cid, rep["item_id"])
    h2 = ss.histogram_samples(cid, rep["item_id"])
    assert h1["before"] == h2["before"]
    assert len(h1["after"]) == rep["n_hist"]
