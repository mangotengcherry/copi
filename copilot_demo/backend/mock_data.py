"""결정론적 합성 mock 데이터 생성. 실행: python -m backend.mock_data"""
import json
import random
import numpy as np

from backend.constants import (
    DATA_DIR, SEED, ITEM_GROUPS, PROCESS_LEVEL1, PROCESS_MODULES, PROCESS_LAYERS,
    CHANGE_TYPES, FINAL_DECISIONS,
)

N_CHANGES = 15
N_ITEMS = 200
N_PAST = 12

_CHANGE_TEMPLATES = [
    ("증착온도 +5%", "recipe_param", "CVD 챔버 증착 온도를 5% 상향 조정하였다."),
    ("가스 유량 감소", "recipe_param", "반응 가스 유량을 8% 감소시켜 두께 균일도를 개선하였다."),
    ("타겟 소재 교체", "material", "메탈 증착 타겟을 신규 벤더 소재로 교체하였다."),
    ("세정 스텝 추가", "route", "식각 전 추가 세정 스텝을 라우트에 삽입하였다."),
    ("챔버 하드웨어 교체", "hardware", "노후 챔버의 샤워헤드를 신규 부품으로 교체하였다."),
    ("이온주입 도즈 상향", "recipe_param", "이온주입 도즈를 3% 상향하여 문턱전압을 조정하였다."),
    ("CMP 압력 변경", "recipe_param", "CMP 연마 압력을 재설정하여 평탄도를 조정하였다."),
    ("포토 노광량 조정", "recipe_param", "노광 에너지를 미세 조정하여 CD를 보정하였다."),
    ("슬러리 벤더 변경", "material", "CMP 슬러리 공급 벤더를 변경하였다."),
    ("확산 시간 단축", "recipe_param", "확산 공정 시간을 단축하여 열예산을 줄였다."),
    ("식각 레시피 변경", "recipe_param", "식각 종료점 검출 조건을 변경하였다."),
    ("배선 라우트 변경", "route", "M1 배선 경로를 재설계하여 저항을 낮추었다."),
    ("설비 이관", "hardware", "동일 공정을 신규 설비로 이관하였다."),
    ("전구체 교체", "material", "증착 전구체를 신규 물질로 교체하였다."),
    ("어닐 온도 변경", "recipe_param", "후속 어닐 온도를 조정하였다."),
]


def benjamini_hochberg(pvals):
    """BH FDR 보정. 입력 순서를 유지한 q-value 리스트 반환."""
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * n / (rank + 1)
        prev = min(prev, val)
        q[i] = min(1.0, prev)
    return q


def _make_changes(rng):
    changes = []
    for k in range(N_CHANGES):
        tmpl = _CHANGE_TEMPLATES[k]
        level1 = PROCESS_LEVEL1[k % len(PROCESS_LEVEL1)]
        level2 = PROCESS_MODULES[k % len(PROCESS_MODULES)]
        level3 = PROCESS_LAYERS[k % len(PROCESS_LAYERS)]
        status = "past" if k < N_PAST else "current"
        month = (k % 10) + 1
        changes.append({
            "change_id": f"CHG-2024-{k+1:03d}",
            "process_step": {
                "level1": level1, "level2": level2, "level3": level3,
                "path": f"{level1}/{level2}/{level3}",
            },
            "change_type": tmpl[1],
            "change_direction": tmpl[0],
            "description_ko": tmpl[2],
            "period_start": f"2024-{month:02d}-01",
            "period_end": f"2024-{month:02d}-20",
            "lot_count": int(rng.integers(20, 61)),
            "status": status,
        })
    return changes


def _make_items(rng):
    items = []
    for i in range(N_ITEMS):
        group = ITEM_GROUPS[i % len(ITEM_GROUPS)]
        items.append({
            "item_id": f"EDS_{i+1:03d}",
            "item_group": group,
            "struct_location": f"{group}_loc_{i % 7}",
            "related_bin": f"BIN{(i % 9) + 1}",
            "baseline_mean": round(float(rng.uniform(0.5, 5.0)), 3),
            "baseline_std": round(float(rng.uniform(0.05, 0.3)), 3),
            "unit": "a.u.",
        })
    return items


def _make_stats_and_ranking(rng, changes, items):
    stats, ranking = [], []

    for c in changes:
        cid = c["change_id"]
        n_signal = int(rng.integers(2, 4))  # 2~3개 group
        signal_groups = list(rng.choice(ITEM_GROUPS, size=n_signal, replace=False))
        signal_dir = {g: ("up" if rng.random() < 0.5 else "down") for g in signal_groups}

        rows, pvals = [], []
        for it in items:
            g = it["item_group"]
            if g in signal_groups:
                mag = float(rng.uniform(0.4, 1.2))
                effect = mag if signal_dir[g] == "up" else -mag
                p = float(rng.uniform(1e-5, 0.02))
                direction = signal_dir[g]
            else:
                effect = float(rng.normal(0.0, 0.1))
                p = float(rng.uniform(0.05, 1.0))
                direction = "none"
            rows.append({"item": it, "effect": effect, "p": p, "direction": direction})
            pvals.append(p)

        qvals = benjamini_hochberg(pvals)

        # 각 group의 대표 item = |effect| 최대
        rep_by_group = {}
        for r, q in zip(rows, qvals):
            g = r["item"]["item_group"]
            if g not in rep_by_group or abs(r["effect"]) > abs(rep_by_group[g]["effect"]):
                rep_by_group[g] = {**r, "q": q}

        for r, q in zip(rows, qvals):
            it = r["item"]
            g = it["item_group"]
            is_rep = rep_by_group[g]["item"]["item_id"] == it["item_id"]
            entry = {
                "change_id": cid, "item_id": it["item_id"], "item_group": g,
                "p_value": round(r["p"], 6), "q_value": round(q, 6),
                "effect_size": round(r["effect"], 4), "shift_direction": r["direction"],
                "is_representative": is_rep,
            }
            if is_rep:
                base = it["baseline_mean"]
                std = it["baseline_std"]
                shift = r["effect"] * std * 2.0
                entry.update({
                    "mean_before": round(base, 4),
                    "std_before": round(std, 4),
                    "mean_after": round(base + shift, 4),
                    "std_after": round(std, 4),
                    "n_hist": 200,
                })
            stats.append(entry)

        # ranking: 8개 group 전부
        for g in ITEM_GROUPS:
            grp_rows = [r for r in rows if r["item"]["item_group"] == g]
            grp_q = [q for r, q in zip(rows, qvals) if r["item"]["item_group"] == g]
            sig_ratio = sum(q < 0.1 for q in grp_q) / max(1, len(grp_q))
            max_eff = max(abs(r["effect"]) for r in grp_rows)
            stat_sig = min(1.0, sum(1 for q in grp_q if q < 0.1) / 3.0)
            eff_mag = min(1.0, max_eff / 1.2)
            coverage = sig_ratio
            hist_rec = float(rng.uniform(0.0, 0.4)) + (0.4 if g in signal_groups else 0.0)
            hist_rec = min(1.0, hist_rec)
            risk = 0.4 * stat_sig + 0.35 * eff_mag + 0.15 * coverage + 0.10 * hist_rec
            level = "High" if risk >= 0.6 else ("Med" if risk >= 0.35 else "Low")
            ranking.append({
                "change_id": cid, "group": g,
                "risk_score": round(risk, 4), "risk_level": level,
                "representative_item": rep_by_group[g]["item"]["item_id"],
                "factors": {
                    "stat_significance": round(stat_sig, 4),
                    "effect_magnitude": round(eff_mag, 4),
                    "coverage": round(coverage, 4),
                    "historical_recurrence": round(hist_rec, 4),
                },
            })
    return stats, ranking


def _make_review_cards(rng, changes, stats):
    cards = []
    sig_by_change = {}
    for s in stats:
        if s["q_value"] < 0.1:
            sig_by_change.setdefault(s["change_id"], {})
            g = s["item_group"]
            sig_by_change[s["change_id"]][g] = s["shift_direction"]
    reviewers = ["김공정", "이설비", "박수율", "최소자"]
    for c in changes:
        if c["status"] != "past":
            continue
        cid = c["change_id"]
        affected = [{"group": g, "direction": d}
                    for g, d in list(sig_by_change.get(cid, {}).items())[:3]]
        decision = FINAL_DECISIONS[rng.integers(0, len(FINAL_DECISIONS))]
        cards.append({
            "change_id": cid,
            "final_decision": decision,
            "affected_groups": affected,
            "confounding_review": "동시 진행된 설비 PM 이력 및 lot mix 편차를 검토함.",
            "follow_up_actions": ["추가 lot 모니터링", "설비 편차 재확인"],
            "engineer_comment": f"{c['change_direction']} 변경 후 {', '.join(a['group'] for a in affected) or '유의 group 없음'} 관찰됨.",
            "reviewer": reviewers[int(rng.integers(0, len(reviewers)))],
            "date": c["period_end"],
        })
    return cards


def generate_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    changes = _make_changes(rng)
    items = _make_items(rng)
    stats, ranking = _make_stats_and_ranking(rng, changes, items)
    cards = _make_review_cards(rng, changes, stats)

    def _write(name, obj):
        (DATA_DIR / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    _write("changes.json", changes)
    _write("eds_items.json", items)
    _write("stats.json", stats)
    _write("ranking.json", ranking)
    _write("review_cards.json", cards)


if __name__ == "__main__":
    generate_all()
    print(f"mock data written to {DATA_DIR}")
