import json
import pytest
from backend import llm_client as llm
from backend.models import CopilotSummary

CONTEXT = {
    "overview": {"change_id": "CHG-2024-013", "change_direction": "설비 이관",
                 "process_path": "FE/Metal/M1", "significant_item_count": 12,
                 "affected_groups": [{"group": "Metal", "direction": "up", "significant_item_count": 8}]},
    "ranking": [{"group": "Metal", "risk_level": "High", "risk_score": 0.72,
                 "representative_item": "EDS_005"}],
    "similar": [{"change_id": "CHG-2024-003", "final_decision": "공정영향", "total_score": 0.61}],
}


def test_fallback_summary_is_valid_schema():
    s = llm.fallback_summary(CONTEXT)
    assert isinstance(s, CopilotSummary)
    assert "Metal" in " ".join(p.group for p in s.priority_groups)


def test_generate_summary_uses_llm_on_success(monkeypatch):
    good = {
        "overview_summary": "Metal group에서 유의한 shift가 관찰됨.",
        "priority_groups": [{"group": "Metal", "reason": "risk High, 유의 item 다수"}],
        "confounding_warnings": ["설비 이관에 따른 편차 가능성"],
        "similar_case_insights": [{"change_id": "CHG-2024-003", "relevance": "동일 유형"}],
        "suggested_checks": ["Metal 대표 item 재확인"],
    }
    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: good)
    s, used = llm.generate_copilot_summary(CONTEXT)
    assert used is True
    assert s.priority_groups[0].group == "Metal"


def test_generate_summary_falls_back_on_persistent_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(llm, "call_ollama", boom)
    s, used = llm.generate_copilot_summary(CONTEXT)
    assert used is False
    assert isinstance(s, CopilotSummary)


def test_generate_summary_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad json")
        return {"overview_summary": "ok", "priority_groups": [],
                "confounding_warnings": [], "similar_case_insights": [],
                "suggested_checks": []}
    monkeypatch.setattr(llm, "call_ollama", flaky)
    s, used = llm.generate_copilot_summary(CONTEXT)
    assert calls["n"] == 2 and used is True
