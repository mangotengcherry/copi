"""Copilot 추천 로직 (전부 rule-based, LLM 미사용).

추천값과 근거는 코드 계산값(ranking risk_level/score, change_type, signal,
유사사례 과거판정)에서 결정론적으로 도출한다. LLM은 이 모듈에 개입하지 않는다.
사용자는 추천을 승인/거부만 한다.
"""
from collections import Counter

from backend import data_store as ds, ranking as ranking_mod, similarity
from backend.stats_summary import build_overview
from backend.constants import CHECKLIST
from backend.models import (
    GroupTagRecommendation, FinalDecisionRecommendation,
    ChecklistRecommendation, RecommendationResponse,
)

TOP_GROUPS = 5          # UI에 노출되는 상위 group 수와 일치
LOT_SUFFICIENT = 30     # 통계 판단에 충분하다고 보는 lot 수 임계


def _group_tags(change, table):
    ct = change["change_type"]
    recs = []
    for r in table[:TOP_GROUPS]:
        if r.risk_level == "High":
            tags = ["중요", "설비의심" if ct == "hardware" else "공정가능"]
        elif r.risk_level == "Med":
            tags = ["follow-up"]
        else:
            tags = ["Noise"]
        reason = f"risk_level {r.risk_level} (score {r.risk_score}), change_type {ct}"
        recs.append(GroupTagRecommendation(
            group=r.group, risk_level=r.risk_level,
            recommended_tags=tags, reason=reason))
    return recs


def _final_decision(change, table, similar):
    ct = change["change_type"]
    high_groups = [r.group for r in table if r.risk_level == "High"]
    decisions = [c.final_decision for c in similar.candidates if c.final_decision != "N/A"]
    if decisions:
        common, cnt = Counter(decisions).most_common(1)[0]
        if cnt >= 2:
            return FinalDecisionRecommendation(
                recommended=common, reason=f"유사사례 {cnt}건이 '{common}' 판정")
    if high_groups:
        joined = ", ".join(high_groups)
        if ct == "hardware":
            return FinalDecisionRecommendation(
                recommended="설비영향", reason=f"High group({joined}) + change_type hardware")
        return FinalDecisionRecommendation(
            recommended="공정영향", reason=f"High group({joined}) 존재, change_type {ct}")
    return FinalDecisionRecommendation(recommended="Noise", reason="High risk group 없음")


def _checklist(change, overview):
    n = change["lot_count"]
    sig = overview.significant_item_count
    recs = []
    for sec in CHECKLIST:
        for i, item in enumerate(sec["items"]):
            auto = False
            if sec["section"] == "물량" and i == 0:
                auto = n >= LOT_SUFFICIENT
                reason = f"적용 lot {n}건" + (" (충분)" if auto else " (부족 가능 — 확인)")
            elif sec["section"] == "물량" and i == 1:
                auto = True
                reason = "before/after 각 200 sample로 물량 균형"
            elif sec["section"] == "통계" and i == 0:
                reason = f"유의 item {sig}개 — 목록 확인 필요"
            else:
                reason = "엔지니어 확인 필요"
            recs.append(ChecklistRecommendation(
                section=sec["section"], index=i, item=item,
                auto_checked=auto, reason=reason))
    return recs


def build_recommendations(cid):
    change = ds.get_change(cid)
    table = ranking_mod.get_ranking_table(cid)
    similar = similarity.find_similar(cid)
    overview = build_overview(cid)
    return RecommendationResponse(
        group_tags=_group_tags(change, table),
        final_decision=_final_decision(change, table, similar),
        checklist=_checklist(change, overview),
    )
