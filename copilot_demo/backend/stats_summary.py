"""overview 통계 요약 + before/after 히스토그램 (모두 코드 계산)."""
import numpy as np

from backend import data_store as ds
from backend.constants import SIGNIFICANCE_Q, ITEM_GROUPS
from backend.models import OverviewResponse, GroupImpact, Change


def significant_signature(cid):
    """q<0.1 유의 item을 group→dominant direction 으로 요약."""
    sig = {}
    dir_count = {}
    for s in ds.get_stats(cid):
        if s["q_value"] < SIGNIFICANCE_Q and s["shift_direction"] in ("up", "down"):
            g = s["item_group"]
            dir_count.setdefault(g, {"up": 0, "down": 0})
            dir_count[g][s["shift_direction"]] += 1
    for g, dc in dir_count.items():
        sig[g] = "up" if dc["up"] >= dc["down"] else "down"
    return sig


def build_overview(cid):
    change = ds.get_change(cid)
    stats = ds.get_stats(cid)
    sig = [s for s in stats if s["q_value"] < SIGNIFICANCE_Q]
    signature = significant_signature(cid)
    counts = {}
    for s in sig:
        counts[s["item_group"]] = counts.get(s["item_group"], 0) + 1
    affected = [
        GroupImpact(group=g, direction=signature[g], significant_item_count=counts.get(g, 0))
        for g in ITEM_GROUPS if g in signature
    ]
    affected.sort(key=lambda x: x.significant_item_count, reverse=True)
    return OverviewResponse(
        change=Change(**change),
        total_items_tested=len(stats),
        significant_item_count=len(sig),
        affected_groups=affected,
    )


def histogram_samples(cid, item_id):
    stats = ds.get_stats(cid)
    row = next((s for s in stats if s["item_id"] == item_id and s["is_representative"]), None)
    if row is None:
        row = next((s for s in stats if s["item_id"] == item_id), None)
    if row is None or row.get("mean_before") is None:
        # 대표 item이 아니면 baseline으로 대체
        item = next(it for it in ds.get_items() if it["item_id"] == item_id)
        mb = ma = item["baseline_mean"]
        sb = sa = item["baseline_std"]
        n = 200
    else:
        mb, sb = row["mean_before"], row["std_before"]
        ma, sa = row["mean_after"], row["std_after"]
        n = row["n_hist"]
    seed = abs(hash(f"{cid}:{item_id}")) % (2**32)
    rng = np.random.default_rng(seed)
    before = [round(float(x), 4) for x in rng.normal(mb, sb, n)]
    after = [round(float(x), 4) for x in rng.normal(ma, sa, n)]
    return {"item_id": item_id, "before": before, "after": after}
