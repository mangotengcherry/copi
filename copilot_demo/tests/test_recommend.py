from backend import recommend, data_store as ds
from backend.constants import FEEDBACK_TAGS, FINAL_DECISIONS, CHECKLIST


def _current_id():
    ds.reload()
    return [c for c in ds.get_changes() if c["status"] == "current"][0]["change_id"]


def test_group_tags_valid_and_bounded():
    rec = recommend.build_recommendations(_current_id())
    assert 1 <= len(rec.group_tags) <= recommend.TOP_GROUPS
    for gt in rec.group_tags:
        assert gt.risk_level in ("High", "Med", "Low")
        assert gt.recommended_tags  # 비어있지 않음
        assert all(t in FEEDBACK_TAGS for t in gt.recommended_tags)


def test_high_group_recommends_important_tag():
    # 현재 change는 signal 구조상 High group을 가진다
    rec = recommend.build_recommendations(_current_id())
    highs = [gt for gt in rec.group_tags if gt.risk_level == "High"]
    assert highs, "expected at least one High group"
    assert all("중요" in gt.recommended_tags for gt in highs)


def test_final_decision_is_valid_choice():
    rec = recommend.build_recommendations(_current_id())
    assert rec.final_decision.recommended in FINAL_DECISIONS
    assert rec.final_decision.reason


def test_checklist_covers_all_items_with_auto_flags():
    rec = recommend.build_recommendations(_current_id())
    total = sum(len(s["items"]) for s in CHECKLIST)
    assert len(rec.checklist) == total
    # 물량 섹션 항목은 auto_checked 판정을 갖는다(자동 검증 가능 항목)
    mullyang = [c for c in rec.checklist if c.section == "물량"]
    assert len(mullyang) == 2
    assert any(c.auto_checked for c in mullyang)
